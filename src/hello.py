"""
Self-contained PyTorch "hello world" training example.

Run from the project root as:

    python -m src.hello
"""

import torch


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, steps=10).unsqueeze(1)
    y = 2 * x + 1
    return x, y


def main() -> None:
    x_train, y_train = make_training_data()

    model = torch.nn.Linear(in_features=1, out_features=1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 200
    for _ in range(num_epochs):
        model.train()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    learned_weight = model.weight.item()
    learned_bias = model.bias.item()

    new_x = torch.tensor([[5.0]])
    model.eval()
    with torch.no_grad():
        predicted_y = model(new_x)

    print("PyTorch Training Demo: y = 2x + 1")
    print("==================================")
    print("Learned weight (should be near 2):", learned_weight)
    print("Learned bias   (should be near 1):", learned_bias)
    print()
    print("Testing on a NEW x value (not in training data):")
    print("Input x:", new_x.item())
    print("Predicted y:", predicted_y.item())
    print("Exact y (2x + 1):", 2 * new_x.item() + 1)


if __name__ == "__main__":
    main()

