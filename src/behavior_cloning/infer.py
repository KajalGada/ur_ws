import torch
from models.policy import BCPolicy
import numpy as np

STATE_DIM = 7
ACTION_DIM = 6

model = BCPolicy(STATE_DIM, ACTION_DIM)
model.load_state_dict(torch.load("bc_ur5_policy.pth"))
model.eval()

def get_action(state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(state_t).squeeze(0).numpy()
    return action
