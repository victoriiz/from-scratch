import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

torch.manual_seed(0)

# practice dataset: y = 2x_1 - 3x_2 + 0.5x_3 + 1
# if our split, shapes, training loop, and optimizer are working, nn.Linear(3, 1) should learn weights near [2, -3, 0.5] and bias close to 1
X = torch.randn(1000, 3)
noise = 0.1 * torch.randn(1000, 1)
y = 2 * X[:, 0:1] - 3 * X[:, 1:2] + 0.5 * X[:, 2:3] + 1 + noise 

EPOCHS = 500

# initialize training and validation datasets
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
    dataset = train_dataset,
    batch_size=32,   # number of samples per batch
    shuffle=True   # reshuffle data at ever epoch
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=32
)

# initialize 3-input model, SGD optimizer, MSE loss function
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

def eval(model, loader):
    """ Determine the average loss for a model on a loader. """
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples

best_val_loss = float('inf')
best_state = None

# training loop
for epoch in range(EPOCHS):
    model.train()

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward() # compute gradients (dloss/dw for params that have requires_grad=True, adds results to each param's .grad attribute)
        optimizer.step() # reads each param's .grad attribute and updates param vals according to optimizer's rule

    val_loss = eval(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | validation MSE: {val_loss:.4f}")

# test
model.load_state_dict(best_state)
test_loss = eval(model, test_loader)

print(f"best validation MSE: {best_val_loss:.4f}")
print(f"test mse: {test_loss:.4f}")
print(f"learned weights: {model.weight.detach()}")
print(f"learned bias: {model.bias.detach()}")
