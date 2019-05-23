from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import models
import pickle
import os
from utils import *

# P and Q not processed by softmax
def kl_divergence(P, Q):
    eps = 1e-10
    softmax = nn.Softmax(dim=1)
    P_prob = softmax(P)
    Q_prob = softmax(Q)
    KL = P_prob*torch.log((P_prob+eps)/(Q_prob+eps))
    KL = KL.sum(dim=1)
    return KL

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("finished" + str(batch_idx))
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return float(correct)/float(len(test_loader.dataset))

def sample_kl(args, model, device, sample_loader):
    model.eval()
    sum_kl = 0
    num_samples = 0
    for batch_idx, (data, target) in enumerate(sample_loader):
        data = data.cuda()
        # filter out to get data
        N = data.shape[0]
        data_start, data_observe = data.split([1, N-1], dim=0)
        target_start,target_observe = target.split([1, N-1], dim=0)

        data_filtered_observe = data_observe[target_observe!=target_start.item()]
        if data_filtered_observe.shape[0] is 0:
            continue

        # compute actual kl divergence
        kl_dist = kl_divergence(model(data_start).expand(data_filtered_observe.shape[0],-1), model(data_filtered_observe)).detach()

        sum_kl += kl_dist.sum().item()
        num_samples += data_filtered_observe.shape[0]
    return sum_kl/num_samples

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-dir', type=str, help='save directory')
    parser.add_argument('--arch', type=str, help='architecture to choose')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/mnt/ceph_fs/public_bliao/datasets/cifar/', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        ),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/mnt/ceph_fs/public_bliao/datasets/cifar/', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        ),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    sample_kl_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/mnt/ceph_fs/public_bliao/datasets/cifar/', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        ),
        batch_size=5, shuffle=False)


    model = models.__dict__[args.arch]().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    all_acc = []
    all_kl = []
    acc = test(args, model, device, test_loader)
    kl = sample_kl(args, model, device, sample_kl_loader)
    all_acc.append(acc)
    all_kl.append(kl)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # calculate standard generalization and kl divergence
        acc = test(args, model, device, test_loader)
        kl = sample_kl(args, model, device, sample_kl_loader)
        print("Acc: %.7f, KL divergence: %.7f"%(acc, kl))

        # save the result
        all_acc.append(acc)
        all_kl.append(kl)

        if (args.save_model):
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_middle.pth"))

    # print the final result
    print(all_acc)
    print(all_kl)
    if (args.save_model):
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))

if __name__ == '__main__':
    main()
