"""
hello.py
========

A very small, self‑contained “Hello World” example using PyTorch.

The goal of this script is to give you a concrete, working example
of how to:

* Import PyTorch.
* Create a tensor (PyTorch’s core data structure).
* Perform a simple mathematical operation.
* Inspect and print the results to the terminal.

You can run this file from the project root (after activating your
virtual environment and installing dependencies) with:

    python3 hello.py
"""

import torch  # Main PyTorch library providing tensors and operations.


def main() -> None:
    """
    Entry point for the Hello World demo.

    Step‑by‑step, this function:

    1. Builds a small 2x2 tensor of floating‑point numbers.
    2. Multiplies the tensor by 2, using PyTorch’s element‑wise math.
    3. Prints the input and the result to the console.
    4. Shows some basic metadata (device and shape) so you can see
       how PyTorch represents tensors internally.
    """

    # Create a 2x2 tensor of floats on the default device (CPU by default).
    #
    # In PyTorch, `torch.tensor(...)` is the fundamental way to construct
    # tensor objects from Python data like lists or tuples.
    # Here, we create:
    #
    #   [[1.0, 2.0],
    #    [3.0, 4.0]]
    #
    # The outer list represents rows; each inner list is a row.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Perform a simple element‑wise operation on the tensor.
    #
    # PyTorch overloads Python’s arithmetic operators. When we write
    #
    #   y = x * 2
    #
    # every element in `x` is multiplied by 2, producing a new tensor `y`.
    # The original tensor `x` is unchanged.
    y = x * 2

    # A simple human‑readable header so you can recognize this script’s output.
    print("PyTorch Hello World")
    print("--------------------")

    # Print the original tensor values.
    # The `tensor(...)` representation comes from PyTorch and shows:
    # * The nested numeric values.
    # * The default dtype (float32) and device (cpu) if they are not obvious.
    print("Input tensor:")
    print(x)

    # Print the tensor after our element‑wise multiplication.
    print("Output tensor (x * 2):")
    print(y)

    # For small experiments, it is often useful to know:
    # * `device`: whether the tensor lives on CPU or GPU.
    # * `shape`: how many dimensions it has and how large they are.
    print("Device:", x.device)
    print("Shape:", x.shape)


# This conditional ensures that `main()` only runs when this file is
# executed as a script, e.g. `python3 hello.py`.
#
# If you import `hello` from another Python file, the code below will
# not run automatically, which makes this module safe to reuse as a
# simple example or starting point.
if __name__ == "__main__":
    main()

