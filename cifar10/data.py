import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def make_loaders(data_dir="./data", batch_size=64, seed=0):
    transform = transform.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # keep split reproducible across runs
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [45000, 5000], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
