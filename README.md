# Smart Crop Management

Recommend a crop from soil values, diagnose a leaf from a photo, and get a fertilizer plan.

| Input | Model | Output |
|---|---|---|
| N, P, K, temperature, humidity, pH, rainfall | Random Forest | crop + confidence |
| the same values | Random Forest | deficiency / stress label + confidence |
| a leaf photo | EfficientNetB0 (fastai) | disease + confidence |
| both | the above | crop, deficiency, disease, fertilizer advice |

Accounts are local and every prediction is saved as a report.

## Requirements

- **Python 3.10 or 3.11** — TensorFlow 2.16 has no wheels for 3.13+
- About 4 GB of disk for TensorFlow + Torch + fastai

```sh
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                 # http://localhost:5000
```

`requirements.txt` does not pin versions. If you want the exact stack this was written against, use the pins from the sibling `AgroVision` repository: `tensorflow==2.16.1`, `torch==2.3.1`, `torchvision==0.18.1`, `fastai==2.7.17`, `scikit-learn==1.6.0`, `numpy==1.26.4`, `pandas==2.1.4`.

Set `SECRET_KEY` in the environment for stable sessions.

## Routes

| Route | What |
|---|---|
| `/signup`, `/login`, `/logout` | accounts |
| `/` | the form |
| `/predict_soil`, `/predict_image`, `/predict_hybrid` | prediction endpoints (POST) |
| `/reports`, `/report/<id>` | saved history |
| `/about`, `/how_to_use` | static pages |

## Files

| Path | What |
|---|---|
| `app.py` | the Flask app: auth, report storage, the three prediction routes |
| `fertilizer_recommendations.py` | turns N, P, K and pH into a written fertilizer plan |
| `models/RandomForest_label.pkl` | crop classifier |
| `models/RandomForest_deficiency.pkl` | deficiency classifier |
| `models/EfficientNetB0_best.pkl` | leaf-disease learner exported by fastai |
| `models/scaler.pkl` | MinMax scaler for the soil features |
| `crop_recommendation.csv` | the soil dataset |
| `templates/`, `static/` | the UI |

SQLite data lives in `instance/`, uploads in `uploads/`; both are created on first run.

## Notes on this repository

- The README used to tell you to run `app_testing.py`, which never existed (`app_testiing.py` was the typo version). `optamized_app.py` is now `app.py` and the old copy is gone.
- A virtualenv (`vbs/`, 860 files) was committed. It is untracked now and gitignored; tracked files went from 882 to 21.
- This repository and `AgroVision` are the **same codebase**. `AgroVision/optamized_app.py` was byte-identical to this one, and this repository is the one that still had `fertilizer_recommendations.py` when AgroVision had lost it. Pick one to maintain.
- `AgroVision` has the longer README, including the model table and the warning that the soil models expect **scaled** features.

## Known limits

- No CSRF protection, no tests, no rate limiting on the prediction endpoints.
- Models are loaded at import time, so startup takes seconds and about 2 GB of RAM.
- The 66 MB of models are committed directly; Git LFS would suit them better.
