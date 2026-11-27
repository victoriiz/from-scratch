import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
import copy

from linreg import train_one_epoch, evaluate, plot_losses

def main():
    torch.manual_seed(0)

    data = load_diabetes(scaled=False)
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

    n = len(X)
    indices = torch.randperm(n)

    train_end, val_end = int(0.70*n), int(0.85*n)

    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # get training mean and stds to scale data
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    X_train = (X_train-mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,  # number of samples per batch
        shuffle=True,  # reshuffle data at every epoch
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
    )
    
    # initialize model, optimizer, loss function
    model = nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    train_losses, val_losses = [], []

    baseline = y_train.mean()
    baseline_mse = ((y_val - baseline)**2).mean().item()

    for epoch in range(200):
        train_one_epoch(model, train_loader, loss_fn, optimizer)
        train_loss = evaluate(model, train_loader, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    # eval best validation checkpoint on test set
    model.load_state_dict(best_state)
    test_mse = evaluate(model, test_loader, loss_fn)
        
    print(f"Best validation MSE: {best_val_loss:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_mse**0.5:.1f}")
    print(f"Learned weights: {model.weight.detach()}")
    print(f"Learned bias: {model.bias.detach()}")

    plot_losses(train_losses, val_losses, baseline_mse)

if __name__ == "__main__":
    main()

