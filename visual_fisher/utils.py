import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pgd_attack import *

def get_mnist_loaders(train_batch_size, test_batch_size):
    import torchvision.datasets as datasets
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, num_workers=4, drop_last=True)
    return train_loader, test_loader

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

def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.cuda()
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

def pgd_attack_mnist_test(model, device, num_steps, step_size, random_start=False, num_classes=10):
    # ensure that the mode is eval
    model.eval()
    test_batch_size=100
    _,test_loader = get_mnist_loaders(train_batch_size=100, test_batch_size=100)

    print(accuracy(model, test_loader))

    attack_strength = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_images = 0
    num_correct_images = np.zeros(len(attack_strength))
    attack_model = LinfPGDAttack(model, num_steps, step_size, random_start)
    softmax = torch.nn.Softmax(dim=1)
    for x,y in test_loader:
        x = x.to(device)
        y = y.to(device)

        target_class = y.cpu().numpy()

        for i in range(len(attack_strength)):
            x_adv = attack_model.perturb(x, y, attack_strength[i])
            pred_adv = model(x_adv)
            predicted_class = np.argmax(pred_adv.cpu().detach().numpy(), axis=1)
            num_correct_images[i] += np.sum(predicted_class == target_class)

        num_images += test_batch_size
        if num_images % 1000 == 0:
            print(num_correct_images/num_images)
    print(num_correct_images/num_images)

