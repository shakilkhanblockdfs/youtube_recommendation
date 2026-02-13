import torch
from torch.utils.data import DataLoader, TensorDataset
from models.candidate_model import CandidateGenerationModel
from models.ranking_model import RankingModel
from utils import load_ranking_data

NUM_VIDEOS = 2000

hist, videos, labels, watch_time, cont = load_ranking_data("data/out/ranking.npy")

cand = CandidateGenerationModel(NUM_VIDEOS)
cand.load_state_dict(torch.load("candidate.pt", map_location="cpu"))
cand.eval()

with torch.no_grad():
    _, user_emb = cand(hist)

ds = TensorDataset(user_emb, videos, cont, labels, watch_time)
dl = DataLoader(ds, batch_size=64, shuffle=True)

ranker = RankingModel(NUM_VIDEOS)
opt = torch.optim.Adam(ranker.parameters(), lr=1e-3)

for epoch in range(5):
    total = 0
    for u, v, c, y, wt in dl:
        logits = ranker(u, v, c)
        weights = torch.where(y == 1, wt, torch.ones_like(wt))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weights)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    print(f"Epoch {epoch}: loss={total/len(dl):.4f}")

torch.save(ranker.state_dict(), "ranking.pt")
print("Saved ranking.pt")
