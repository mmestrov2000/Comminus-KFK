
import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from facade_defect_model import FacadeElementClassifier, Sam3Segmenter, FacadeDefectDetector

st.set_page_config(page_title="Facade Defect Detector", layout="wide")
st.title("Facade Defect Detector")
st.caption("Single image → immediate report. ZIP upload → CSV report for all images inside.")

with st.sidebar:
    st.header("Model settings")
    yolo_weights = st.text_input("YOLO weights path", value=os.environ.get("YOLO_WEIGHTS", "facade_yolov8n_best.pt"))
    sam3_name = st.text_input("SAM3 model name", value=os.environ.get("SAM3_MODEL", "facebook/sam3"))
    clf_conf = st.slider("YOLO conf", 0.05, 0.9, 0.25, 0.05)
    clf_imgsz = st.selectbox("YOLO imgsz", options=[256, 320, 384, 512, 640], index=2)
    include_debug = st.checkbox("Include debug logs", value=False)
    show_overlay = st.checkbox("Show overlay", value=True)

@st.cache_resource(show_spinner=True)
def load_detector(weights_path: str, sam3_model: str, conf: float, imgsz: int):
    detector = FacadeDefectDetector(
        classifier=FacadeElementClassifier(weights_path=weights_path, conf=conf, imgsz=imgsz),
        segmenter=Sam3Segmenter(model_name=sam3_model),
    ).load()
    return detector

detector = load_detector(yolo_weights, sam3_name, clf_conf, clf_imgsz)

tab1, tab2 = st.tabs(["Single image", "Batch (ZIP)"])

with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(img, caption="Input", use_container_width=True)

        with st.spinner("Running..."):
            inf = detector.infer(img, debug=include_debug)

        with col2:
            st.text_area("Report", detector.solution_report(uploaded.name, inf), height=140)
            if include_debug and "debug" in inf:
                st.text_area("Debug log", "\n".join(inf["debug"]), height=260)

        if show_overlay:
            st.image(detector.visualization(img, inf), caption="Defects highlighted", use_container_width=True)

with tab2:
    zip_up = st.file_uploader("Upload a ZIP containing images", type=["zip"])
    if zip_up is not None:
        zbytes = zip_up.read()
        z = zipfile.ZipFile(io.BytesIO(zbytes))

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        members = [m for m in z.namelist() if Path(m).suffix.lower() in img_exts and not m.endswith("/")]

        if not members:
            st.error("No images found in the ZIP.")
        else:
            st.success(f"Found {len(members)} images.")
            rows = []
            previews = []

            with st.spinner("Processing images..."):
                for m in members:
                    data = z.read(m)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    inf = detector.infer(img, debug=include_debug)

                    row = {
                        "image": m,
                        "status": inf.get("status"),
                        "defects": detector.defects_as_string(inf),
                        "element_raw": inf.get("element_raw"),
                        "element": inf.get("element"),
                        "element_score": inf.get("element_score"),
                    }
                    if include_debug and "debug" in inf:
                        row["debug"] = "\n".join(inf["debug"])
                    rows.append(row)

                    if len(previews) < 6:
                        previews.append((m, img, inf))

            df = pd.DataFrame(rows).sort_values("image")
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV report", data=csv_bytes, file_name="facade_defect_report.csv", mime="text/csv")

            st.markdown("### Preview (first few)")
            for name, img, inf in previews:
                with st.expander(name):
                    st.text(detector.solution_report(name, inf))
                    if show_overlay:
                        st.image(detector.visualization(img, inf), use_container_width=True)
                    if include_debug and "debug" in inf:
                        st.code("\n".join(inf["debug"]))
