import torch
from torch.utils.data import DataLoader, TensorDataset
from models.candidate_model import CandidateGenerationModel
from utils import load_candidate_data

NUM_VIDEOS = 2000

hist, labels = load_candidate_data("data/out/candidate.npy")
ds = TensorDataset(hist, labels)
dl = DataLoader(ds, batch_size=64, shuffle=True)

model = CandidateGenerationModel(NUM_VIDEOS)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total = 0
    for h, y in dl:
        logits, _ = model(h)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch}: loss={total/len(dl):.4f}")

torch.save(model.state_dict(), "candidate.pt")
print("Saved candidate.pt")
