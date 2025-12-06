"""
hello.py
========

A small, self‑contained PyTorch example that *trains* a tiny model.

Instead of just doing a fixed calculation like `x * 2`, this script:

* Creates a synthetic training dataset for the function y = 2x + 1.
* Builds a simple neural network with one linear layer.
* Trains it using gradient descent to approximate that function.
* Uses the trained model to predict y for a **new** x value that
  was not in the original training data.

This is the “training” part you were asking about: we adjust model
parameters (weight and bias) so that the model generalizes to new
inputs, not just ones it has already seen.

Run this file from the project root (after activating your virtual
environment and installing dependencies) with:

    python3 hello.py
"""

import torch  # Main PyTorch library providing tensors, layers, and optimizers.


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a tiny synthetic dataset for the function y = 2x + 1.

    We create ten evenly spaced x values between -1 and 1, then
    compute the corresponding y values using the exact rule.

    Both x and y are returned as 2‑D tensors with shape (N, 1),
    where N is the number of samples. This shape matches what
    `nn.Linear` expects: (batch_size, num_features).
    """

    # Create 10 points between -1.0 and 1.0, inclusive.
    # `linspace` returns a 1‑D tensor of shape (10,).
    x = torch.linspace(-1.0, 1.0, steps=10)

    # Rearrange to shape (10, 1) so each row is a single training example.
    x = x.unsqueeze(1)

    # Our target rule: y = 2x + 1
    # This gives us the "labels" the model should try to match.
    y = 2 * x + 1

    return x, y


def main() -> None:
    """
    Train a simple linear model on y = 2x + 1 and test it.

    High‑level steps:

    1. Create training data tensors (inputs x and targets y).
    2. Define a small model: a single `nn.Linear` layer.
    3. Choose a loss function (MSE) and optimizer (SGD).
    4. Run several training epochs to adjust the model’s parameters.
    5. Use the trained model to predict y for a new x value.
    """

    # 1. Build our synthetic training dataset.
    x_train, y_train = make_training_data()

    # 2. Define a model with one learnable weight and one bias.
    #
    #    nn.Linear(in_features=1, out_features=1)
    #
    # The model computes:
    #
    #    y_hat = weight * x + bias
    #
    # where `weight` and `bias` are parameters stored internally
    # as tensors that PyTorch will update during training.
    model = torch.nn.Linear(in_features=1, out_features=1)

    # 3. Define a loss function and optimizer.
    #
    # We use Mean Squared Error (MSE) because this is a simple
    # regression problem, and Stochastic Gradient Descent (SGD)
    # to update the model’s parameters.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 4. Training loop.
    #
    # We will do a small number of epochs because the problem
    # is tiny and easy to fit.
    num_epochs = 200

    for epoch in range(num_epochs):
        # Put the model in training mode (good habit, though it
        # mainly matters for layers like dropout or batch norm).
        model.train()

        # Forward pass: compute predictions for all training inputs.
        y_pred = model(x_train)

        # Compute how far off we are from the true targets.
        loss = loss_fn(y_pred, y_train)

        # Reset gradients from the previous step.
        optimizer.zero_grad()

        # Backward pass: compute gradients of loss w.r.t. parameters.
        loss.backward()

        # Parameter update: move weight and bias a small step in the
        # direction that reduces the loss.
        optimizer.step()

    # 5. After training, we can inspect the learned parameters.
    learned_weight = model.weight.item()
    learned_bias = model.bias.item()

    # And we can test the model on a new input that it never saw
    # during training, for example x = 5.0.
    new_x = torch.tensor([[5.0]])
    model.eval()  # Switch to evaluation mode (again, a good habit).
    with torch.no_grad():  # Disable gradient tracking for inference.
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


# This conditional ensures that `main()` only runs when this file is
# executed as a script, e.g. `python3 hello.py`.
#
# If you import `hello` from another Python file, the code below will
# not run automatically, which makes this module safe to reuse as a
# simple example or starting point.
if __name__ == "__main__":
    main()

