Place the trained `descriptor.onnx` here (from the Kaggle notebook's Output
tab — 7 cells in README/DEPLOYMENT.md). It is auto-detected at import time:

    backend/models/descriptor.onnx   (1-5 MB, opset 17, input "patch" 1x1x128x128)

Restart uvicorn after dropping it in, then verify:
    curl http://localhost:8000/health  ->  "learned_model_loaded": true

Validation gate before shipping: the notebook's Cell 6 must print triplet
ranking accuracy > 0.5 (ideally > 0.9). This directory is gitignored.
