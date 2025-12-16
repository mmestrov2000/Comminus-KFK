# Facade Defect Detector (Streamlit)

## Features
- Single image upload → immediate report (+ optional overlay + optional debug log)
- Batch upload via ZIP → CSV report (one row per image)

## Run
1) Install deps:
```bash
pip install -r requirements.txt
```

2) (If needed) login to Hugging Face for SAM3:
```bash
huggingface-cli login
```

3) Run the app:
```bash
export YOLO_WEIGHTS="/absolute/path/to/facade_yolov8n_best.pt"
streamlit run app.py
```

You can also set the weights path in the sidebar.

## Batch mode
Zip your test folder (subfolders ok), upload the ZIP, then download the CSV.
