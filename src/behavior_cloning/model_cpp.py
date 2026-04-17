import torch

model = torch.nn.Sequential(
    torch.nn.Linear(6, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 6)
)
model.load_state_dict(torch.load("bc_ur5_policy.pth"))
model.eval()

example_input = torch.randn(6)

traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.ts")