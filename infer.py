import torch
from models.candidate_model import CandidateGenerationModel
from models.ranking_model import RankingModel

NUM_VIDEOS = 2000
TOP_K = 50
FINAL_K = 5

cand = CandidateGenerationModel(NUM_VIDEOS)
cand.load_state_dict(torch.load("candidate.pt", map_location="cpu"))
cand.eval()

ranker = RankingModel(NUM_VIDEOS)
ranker.load_state_dict(torch.load("ranking.pt", map_location="cpu"))
ranker.eval()

# Fake user history
watched = torch.randint(0, NUM_VIDEOS, (1, 10))

with torch.no_grad():
    logits, user_emb = cand(watched)
    topk = torch.topk(logits, k=TOP_K, dim=-1).indices[0]

    user_emb = user_emb.repeat(TOP_K, 1)
    cont = torch.rand(TOP_K, 2)  # dummy features

    scores = ranker(user_emb, topk, cont)
    final = torch.topk(scores, k=FINAL_K)

print("Recommended video IDs:", topk[final.indices].tolist())
