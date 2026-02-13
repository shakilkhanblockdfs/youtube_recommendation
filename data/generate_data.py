import numpy as np
import os
import random

np.random.seed(42)
random.seed(42)

NUM_USERS = 5000
NUM_VIDEOS = 2000
HISTORY_LEN = 10

os.makedirs("data/out", exist_ok=True)

candidate_data = []
ranking_data = []

for user_id in range(NUM_USERS):
    history = np.random.randint(0, NUM_VIDEOS, size=HISTORY_LEN).tolist()
    next_video = np.random.randint(0, NUM_VIDEOS)

    candidate_data.append({
        "history": history,
        "label": next_video
    })

    # Ranking samples: 1 positive + 4 negatives
    pos_watch_time = np.random.uniform(10, 300)
    ranking_data.append({
        "history": history,
        "video": next_video,
        "label": 1,
        "watch_time": pos_watch_time,
        "time_since_last": np.random.uniform(0, 7),
        "prev_impressions": np.random.randint(0, 20)
    })

    for _ in range(4):
        neg_video = np.random.randint(0, NUM_VIDEOS)
        ranking_data.append({
            "history": history,
            "video": neg_video,
            "label": 0,
            "watch_time": 1.0,
            "time_since_last": np.random.uniform(0, 7),
            "prev_impressions": np.random.randint(0, 20)
        })

np.save("data/out/candidate.npy", candidate_data, allow_pickle=True)
np.save("data/out/ranking.npy", ranking_data, allow_pickle=True)

print("Data generated in data/out/")
