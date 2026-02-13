import torch
import torch.nn as nn

class RankingModel(nn.Module):
    def __init__(self, num_videos, embed_dim=64, hidden_dims=[256,128,64], num_cont_features=2):
        super().__init__()
        self.video_embedding = nn.Embedding(num_videos, embed_dim)

        dim = embed_dim * 2 + num_cont_features
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, user_emb, video_ids, cont_features):
        video_emb = self.video_embedding(video_ids)
        x = torch.cat([user_emb, video_emb, cont_features], dim=1)
        h = self.mlp(x)
        return self.out(h).squeeze(-1)
