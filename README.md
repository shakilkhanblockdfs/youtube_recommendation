# YouTube-style Two-Stage Recommender (Production-style Demo)

## Structure
yt_reco/
├── data/
│   ├── generate_data.py
│   └── out/ (generated)
├── models/
│   ├── candidate_model.py
│   └── ranking_model.py
├── utils.py
├── train_candidate.py
├── train_ranking.py
├── infer.py
├── requirements.txt
└── README.md

## Setup
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

## Run (IN THIS ORDER)
python data/generate_data.py
python train_candidate.py
python train_ranking.py
python infer.py

You should see recommended video IDs printed at the end.
