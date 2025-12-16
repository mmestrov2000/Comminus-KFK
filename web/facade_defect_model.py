
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import torch
from PIL import Image

from ultralytics import YOLO
from transformers import Sam3Processor, Sam3Model


def ensure_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def overlay_binary_masks(image: Image.Image, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    """Overlay list of HxW boolean masks on image (RGBA compositing)."""
    if not masks:
        return image

    rgba = image.convert("RGBA")
    w, h = rgba.size

    # Simple deterministic color cycle
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    for i, m in enumerate(masks):
        m = (m > 0).astype(np.uint8) * 255
        color = colors[i % len(colors)]
        overlay = Image.new("RGBA", (w, h), color + (0,))
        mask_img = Image.fromarray(m, mode="L")
        a = mask_img.point(lambda v: int((v / 255) * 255 * alpha))
        overlay.putalpha(a)
        rgba = Image.alpha_composite(rgba, overlay)

    return rgba.convert("RGB")


@dataclass
class FacadeElementClassifier:
    weights_path: str
    device: Optional[str] = None
    conf: float = 0.25
    imgsz: int = 384
    _model: Any = None

    def load(self):
        self.device = self.device or (0 if torch.cuda.is_available() else "cpu")
        self._model = YOLO(self.weights_path)
        return self

    def predict_top_label(self, image: Image.Image) -> (Optional[str], float):
        if self._model is None:
            raise RuntimeError("Classifier not loaded. Call .load() first.")

        results = self._model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None, 0.0

        confs = r.boxes.conf.detach().cpu().numpy()
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
        best_i = int(np.argmax(confs))
        cls_id = int(cls_ids[best_i])
        score = float(confs[best_i])
        name = r.names.get(cls_id, str(cls_id))
        return name, score


@dataclass
class Sam3Segmenter:
    model_name: str = "facebook/sam3"
    device: Optional[str] = None
    threshold: float = 0.6
    mask_threshold: float = 0.5
    _model: Any = None
    _processor: Any = None

    def load(self):
        self.device = self.device or ensure_device()
        self._model = Sam3Model.from_pretrained(self.model_name).to(self.device)
        self._processor = Sam3Processor.from_pretrained(self.model_name)
        self._model.eval()
        return self

    @torch.no_grad()
    def segment(
        self,
        image: Image.Image,
        prompt: str,
        *,
        threshold: Optional[float] = None,
        mask_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("SAM3 not loaded. Call .load() first.")

        thr = self.threshold if threshold is None else float(threshold)
        mthr = self.mask_threshold if mask_threshold is None else float(mask_threshold)

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)

        res = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=thr,
            mask_threshold=mthr,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        out: Dict[str, Any] = {"prompt": prompt, "threshold": thr, "mask_threshold": mthr}
        for k, v in res.items():
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
        return out


@dataclass
class FacadeDefectDetector:
    classifier: FacadeElementClassifier
    segmenter: Sam3Segmenter

    @staticmethod
    def _normalize_element_label(label: Optional[Any]) -> Optional[str]:
        if label is None:
            return None
        if isinstance(label, (list, tuple, dict)) and len(label) == 0:
            return None
        if isinstance(label, (list, tuple)) and len(label) > 0:
            label = label[0]

        s = str(label).strip().lower()
        if s in ("", "none", "null"):
            return None

        direct_map = {
            "facade_element0": "0",
            "facade_element1": "1",
            "facade_element2": "2",
        }
        if s in direct_map:
            return direct_map[s]

        if s.isdigit():
            return s

        m = re.search(r"(?:facade[_\-\s]*element|element)\s*[_\-\s]*([0-9]+)", s)
        if m:
            return m.group(1)

        m = re.search(r"(\d+)(?!.*\d)", s)
        if m:
            return m.group(1)

        return None

    @staticmethod
    def _sam_count(
        image: Image.Image,
        sam: Sam3Segmenter,
        prompt: str,
        *,
        threshold: Optional[float] = None,
        mask_threshold: Optional[float] = None,
        debug: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        used_thr = sam.threshold if threshold is None else float(threshold)
        used_mthr = sam.mask_threshold if mask_threshold is None else float(mask_threshold)

        if debug is not None:
            debug.append(f"[SAM3] prompt='{prompt}' threshold={used_thr} mask_threshold={used_mthr}")

        res = sam.segment(image, prompt, threshold=used_thr, mask_threshold=used_mthr)

        masks = res.get("masks", None)
        scores = res.get("scores", None)

        if masks is None or len(masks) == 0:
            if debug is not None:
                debug.append("[SAM3] -> count=0 (no masks returned)")
            return {
                "count": 0,
                "masks": np.zeros((0, image.size[1], image.size[0]), dtype=bool),
                "scores": np.zeros((0,), dtype=float),
                "prompt": prompt,
                "threshold": used_thr,
                "mask_threshold": used_mthr,
            }

        masks_np = masks.detach().cpu().numpy().astype(bool)
        scores_np = scores.detach().cpu().numpy() if scores is not None else np.ones((masks_np.shape[0],), dtype=float)

        if debug is not None:
            top = np.sort(scores_np)[::-1][:5]
            debug.append(f"[SAM3] -> count={masks_np.shape[0]} top_scores={np.round(top, 4).tolist()}")

        return {
            "count": int(masks_np.shape[0]),
            "masks": masks_np,
            "scores": scores_np,
            "prompt": prompt,
            "threshold": used_thr,
            "mask_threshold": used_mthr,
        }

    def load(self):
        self.classifier.load()
        self.segmenter.load()
        return self

    def infer(self, image: Image.Image, *, debug: bool = False) -> Dict[str, Any]:
        debug_lines: Optional[List[str]] = [] if debug else None

        raw_label, element_score = self.classifier.predict_top_label(image)
        element = self._normalize_element_label(raw_label)

        if debug_lines is not None:
            debug_lines.append(f"[YOLO] raw_label={raw_label!r} normalized={element!r} score={element_score:.4f}")

        defects: List[Dict[str, Any]] = []

        def add_defect(name: str, prompt: str, mask: Optional[np.ndarray] = None, score: Optional[float] = None):
            defects.append({"defect": name, "prompt": prompt, "score": None if score is None else float(score), "mask": mask})
            if debug_lines is not None:
                debug_lines.append(f"[DEFECT] + {name} (from prompt='{prompt}')")

        # ---- Your rules ----
        if element in (None, "0"):
            if debug_lines is not None:
                debug_lines.append("[RULE] element is 0 or None -> 'small circle hole' thr=0.4; if count==1 => 'screw missing'")
            r = self._sam_count(image, self.segmenter, "small circle hole", threshold=0.4, debug=debug_lines)
            if r["count"] == 1:
                i = int(np.argmax(r["scores"])) if r["scores"].size else 0
                add_defect("screw missing", r["prompt"], mask=r["masks"][i], score=float(r["scores"][i]) if r["scores"].size else None)

        elif element == "1":
            if debug_lines is not None:
                debug_lines.append("[RULE] element is 1 -> orange area (<2), orange divisor (==0), black curve (>=1), cracked glass (thr=0.95, >=1)")

            r_orange = self._sam_count(image, self.segmenter, "orange area", threshold=0.25, debug=debug_lines)
            if r_orange["count"] < 2:
                missing = 2 - r_orange["count"]
                add_defect(f"{missing} orange metal sheet missing", r_orange["prompt"])

            r_div = self._sam_count(image, self.segmenter, "orange divisor", threshold=0.2, debug=debug_lines)
            if r_div["count"] == 0:
                add_defect("orange divisor missing", r_div["prompt"])

            r_black = self._sam_count(image, self.segmenter, "black curve", debug=debug_lines)
            if r_black["count"] >= 1:
                i = int(np.argmax(r_black["scores"])) if r_black["scores"].size else 0
                add_defect("seal broken", r_black["prompt"], mask=r_black["masks"][i], score=float(r_black["scores"][i]) if r_black["scores"].size else None)

            r_glass = self._sam_count(image, self.segmenter, "a cracked tempered glass panel", threshold=0.95, debug=debug_lines)
            if r_glass["count"] >= 1:
                i = int(np.argmax(r_glass["scores"])) if r_glass["scores"].size else 0
                add_defect("broken glass", r_glass["prompt"], mask=r_glass["masks"][i], score=float(r_glass["scores"][i]) if r_glass["scores"].size else None)

        elif element == "2":
            if debug_lines is not None:
                debug_lines.append("[RULE] element is 2 -> 'red area'; if count==0 => 'red metal sheet missing'")
            r_red = self._sam_count(image, self.segmenter, "red area", debug=debug_lines)
            if r_red["count"] == 0:
                add_defect("red metal sheet missing", r_red["prompt"])

        status = "FAIL" if defects else "PASS"

        out: Dict[str, Any] = {
            "element": element,
            "element_raw": raw_label,
            "element_score": float(element_score),
            "status": status,
            "defects": defects,
        }
        if debug_lines is not None:
            out["debug"] = debug_lines
        return out

    @staticmethod
    def solution_report(image_name: str, inference: Dict[str, Any]) -> str:
        defects_list = [d["defect"] for d in inference.get("defects", [])]
        defects_str = ", ".join(defects_list) if defects_list else ""
        return f"Image: {image_name}\nStatus: {inference.get('status','FAIL')}\nDefects: {defects_str}"

    @staticmethod
    def defects_as_string(inference: Dict[str, Any]) -> str:
        defects_list = [d["defect"] for d in inference.get("defects", [])]
        return ", ".join(defects_list) if defects_list else ""

    @staticmethod
    def visualization(image: Image.Image, inference: Dict[str, Any]) -> Image.Image:
        masks = [d["mask"] for d in inference.get("defects", []) if d.get("mask") is not None]
        return overlay_binary_masks(image, masks)
