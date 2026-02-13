import torch
import numpy as np

def load_candidate_data(path):
    data = np.load(path, allow_pickle=True)
    histories = torch.tensor([d["history"] for d in data], dtype=torch.long)
    labels = torch.tensor([d["label"] for d in data], dtype=torch.long)
    return histories, labels

def load_ranking_data(path):
    data = np.load(path, allow_pickle=True)
    histories = torch.tensor([d["history"] for d in data], dtype=torch.long)
    videos = torch.tensor([d["video"] for d in data], dtype=torch.long)
    labels = torch.tensor([d["label"] for d in data], dtype=torch.float32)
    watch_time = torch.tensor([d["watch_time"] for d in data], dtype=torch.float32)
    cont = torch.tensor([[d["time_since_last"], d["prev_impressions"]] for d in data], dtype=torch.float32)
    return histories, videos, labels, watch_time, cont
