import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pgd_attack import *

def get_full_cifar10_train_loaders(train_batch_size=128):
    print("loadining training data")
    import torchvision.datasets as datasets
    train_dataset = datasets.CIFAR10(
        root='/mnt/ceph_fs/public_bliao/datasets/cifar/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, drop_last=True)
    return trainloader

def get_semi_cifar10_train_loaders(train_batch_size=100, semi_batch_size=150):
    print("loadining training data")
    import datasets
    train_dataset = datasets.CIFAR10(
        root='/mnt/ceph_fs/public_bliao/datasets/cifar/',
        train=True,
        semi=False,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, drop_last=True)

    semi_dataset = datasets.CIFAR10(
        root='/mnt/ceph_fs/public_bliao/datasets/cifar/',
        train=False,
        semi=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    semiloader = torch.utils.data.DataLoader(semi_dataset, batch_size=semi_batch_size, shuffle=True, num_workers=16, drop_last=True)
    return trainloader, semiloader

def get_cifar10_test_loaders(test_batch_size=100, shuffle=True):
    print("loadining testing data")
    import torchvision.datasets as datasets
    test_dataset = datasets.CIFAR10(
        root='/mnt/ceph_fs/public_bliao/datasets/cifar',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle, num_workers=16, drop_last=True)
    return testloader

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader, device):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        target_class = y.numpy()
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def pgd_attack(model, x, y, num_steps, step_size, random_start=True, num_classes=10, epsilon=8.0/255):
    attack_model = LinfPGDAttack(model, num_steps, step_size, random_start)
    x_adv = attack_model.perturb(x, y, epsilon)
    return x_adv, y

def fisher_solver(model, x, num_steps, step_size, num_classes=10, epsilon=8.0/255):
    solver = LinfFisherSolver(model, num_steps, step_size)
    x_fisher = solver.solve(x,epsilon)
    return x_fisher
