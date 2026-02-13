import torch
import torch.nn as nn

class CandidateGenerationModel(nn.Module):
    def __init__(self, num_videos, embed_dim=64, hidden_dims=[256, 128]):
        super().__init__()
        self.video_embedding = nn.Embedding(num_videos, embed_dim)

        layers = []
        dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h

        self.mlp = nn.Sequential(*layers)
        self.user_proj = nn.Linear(dim, embed_dim)

    def forward(self, watched_ids):
        emb = self.video_embedding(watched_ids)   # (B, T, D)
        user_vec = emb.mean(dim=1)                 # (B, D)
        h = self.mlp(user_vec)
        user_emb = self.user_proj(h)               # (B, D)

        logits = torch.matmul(user_emb, self.video_embedding.weight.t())
        return logits, user_emb
