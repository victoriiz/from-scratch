import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()  # compute gradients and add them to each parameter's .grad
        optimizer.step()  # use .grad to update the parameter values


def evaluate(model, data_loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():  # disable gradient tracking during evaluation
        for X_batch, y_batch in data_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)  # mean loss for this batch

            total_loss += loss.item() * X_batch.size(0)
            total_examples += X_batch.size(0)

    return total_loss / total_examples

def plot_losses(train_losses, val_losses, baseline_mse):
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Training MSE")
    plt.plot(epochs, val_losses, label="Validation MSE")
    plt.axhline(
        baseline_mse,
        color="gray",
        linestyle="--",
        label="Mean-predictor baseline"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Regression learning curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    torch.manual_seed(0)
    X = torch.randn(1000, 3)

    # practice dataset: y = 2x_1 - 3x_2 + 0.5x_3 + 1.
    # If the pipeline works, nn.Linear(3, 1) should learn weights near
    # [2, -3, 0.5] and a bias close to 1.
    noise = 0.1 * torch.randn(1000, 1)
    y = 2 * X[:, 0:1] - 3 * X[:, 1:2] + 0.5 * X[:, 2:3] + 1 + noise

    EPOCHS = 500

    # initialize training, validation, and test datasets
    indices = torch.randperm(1000)

    train_idx = indices[:800]
    val_idx = indices[800:900]
    test_idx = indices[900:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,  # number of samples per batch
        shuffle=True,  # reshuffle data at every epoch
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
    )

    # initialize 3-input model, SGD optimizer, and MSE loss function.
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d} | validation MSE: {val_loss:.4f}")

    # eval best validation checkpoint on test set
    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, loss_fn)

    print(f"Best validation MSE: {best_val_loss:.4f}")
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Learned weights: {model.weight.detach()}")
    print(f"Learned bias: {model.bias.detach()}")


if __name__ == "__main__":
    main()