import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob

from models.policy import BCPolicy
from data.dataset import UR5Dataset

STATE_DIM = 6
ACTION_DIM = 6
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
files = glob.glob("dataset/*.npz")
dataset = UR5Dataset(files)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = BCPolicy(STATE_DIM, ACTION_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0

    for states, actions in loader:
        states = states.to(device)
        actions = actions.to(device)

        preds = model(states)
        loss = loss_fn(preds, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: {total_loss / len(loader)}")

torch.save(model.state_dict(), "bc_ur5_policy.pth")
