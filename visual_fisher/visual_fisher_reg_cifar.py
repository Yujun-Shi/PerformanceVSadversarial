from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import models
import pickle
from utils import *
import os

# output not processed by log_softmax
def visual_fisher(model, device, loader, num_classes):
    model.eval()
    softmax = nn.Softmax(dim=1)
    num_batches = 0
    Avg_F_norm = 0
    for batch_idx, (data,target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad_(True)
        N = data.shape[0]
        logits = model(data)
        prob = softmax(logits)
        log_prob = F.log_softmax(logits, dim=1)

        batch_fisher = torch.zeros(N, 3*32*32, 3*32*32).type(torch.cuda.FloatTensor)
        for i in range(num_classes):
            up_deri = torch.zeros(N, num_classes).type(torch.cuda.FloatTensor)
            up_deri[:,i] = 1
            deri_data_i = torch.autograd.grad(log_prob, data, grad_outputs=up_deri, retain_graph=True)[0]
            deri_data_i = deri_data_i.contiguous().view(N, -1).unsqueeze(dim=1)
            batch_fisher += torch.bmm(deri_data_i.permute(0,2,1), deri_data_i).detach()*prob[:,i].view(N,1,1).detach()
        batch_F_norm = torch.sqrt((batch_fisher*batch_fisher).sum(dim=[1,2])).mean(dim=0).detach().item()
        Avg_F_norm += batch_F_norm
        num_batches += 1
    Avg_F_norm /= num_batches
    return Avg_F_norm

def kl_divergence(P, Q):
    eps = 1e-8
    softmax = nn.Softmax(dim=1)
    P_prob = softmax(P)
    Q_prob = softmax(Q)
    KL = P_prob*torch.log((P_prob+eps)/(Q_prob+eps))
    KL = KL.sum(dim=1).mean()
    return KL

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data_fisher = fisher_solver(model, data, num_steps=10, step_size=2.0/255)

        optimizer.zero_grad()
        output = model(data)
        output_fisher = model(data_fisher)

        loss = criterion(output, target)
        loss_fisher = kl_divergence(output, output_fisher)

        loss = loss + 5*loss_fisher
        loss.backward()
        optimizer.step()
        if batch_idx%10 == 0:
            print('finished '+str(batch_idx))
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

def test_adv(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().cuda()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv, target = pgd_attack(model, data, target, num_steps=20, step_size=2.0/255, random_start=False)
        output_adv = model(data_adv)

        test_loss += criterion(output_adv, target).item() # sum up batch loss
        pred = output_adv.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nAdv Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

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

    model = models.__dict__[args.arch]().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    all_F_fisher = []
    all_adv_acc = []
    F_fisher = visual_fisher(model, device, test_loader, 10)
    all_F_fisher.append(F_fisher)
    print("average F norm: %.7f", F_fisher)
    for epoch in range(1, args.epochs + 1): 
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

        # store adversarial accuracy @ this epoch
        adv_acc = test_adv(args, model, device, test_loader)
        all_adv_acc.append(adv_acc)

        # store F norm of fisher information @ this epoch
        F_fisher = visual_fisher(model, device, test_loader, 10)
        all_F_fisher.append(F_fisher)
        print("F norm at this epoch: %.7f", F_fisher)
        if (args.save_model):
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_middle.pth"))

    print(all_adv_acc)
    with open('fisher_res/adv_acc_'+args.arch+'.p', 'wb') as f:
        pickle.dump(all_adv_acc, f)
    print(all_F_fisher)
    with open('fisher_res/fisher_norm_'+args.arch+'.p', 'wb') as f:
        pickle.dump(all_F_fisher, f)
    if (args.save_model):
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))

if __name__ == '__main__':
    main()
