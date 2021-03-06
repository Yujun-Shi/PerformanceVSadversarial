import argparse
import os
import time
import shutil
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import sys
sys.path.insert(0, '..')
import models

from utils import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--save-dir', default='', type=str)
parser.add_argument('--fisher-steps', default=10, type=int, help='num of step of fisher')
parser.add_argument('--fisher-stepsize', default=2.0/255, type=float, help='stepsize of fisher')
parser.add_argument('--mult-coef', default=1.0, type=float)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
best_prec = 0

def main():
    global args, best_prec
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print("=> creating model '{}'".format(args.arch))
    if use_gpu:
        if args.cifar_type == 10:
            model = models.__dict__[args.arch](num_classes=10)
        else:
            model = models.__dict__[args.arch](num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        fdir = args.save_dir
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        model = nn.DataParallel(model).cuda()
        #model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        fisher_train_loader = get_full_cifar10_train_loaders(train_batch_size=args.batch_size) 
        test_loader = get_cifar10_test_loaders()
        attack_test_loader = get_cifar10_test_loaders()
    else:
        raise NotImplementedError('not implemented this one yet')

    print(model)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        fisher_train(fisher_train_loader, model, criterion, optimizer, epoch)
        # evaluate on test set
        prec, prec_attack = validate_normal_and_attack(test_loader, attack_test_loader, model, criterion)

        # remember best precision and save checkpoint
        if (epoch + 1)%10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec': prec,
                'optimizer': optimizer.state_dict(),
            }, fdir, 'checkpoint' + str(epoch+1) + '.pth')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def kl_divergence(P, Q):
    eps = 1e-8
    softmax = nn.Softmax(dim=1)
    P_prob = softmax(P)
    Q_prob = softmax(Q)
    KL = P_prob*torch.log((P_prob+eps)/(Q_prob+eps))
    KL = KL.sum(dim=1).mean()
    return KL

def fisher_train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_fisher = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input_orig, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        input_orig, target = input_orig.cuda(), target.cuda()
        input_fisher = fisher_solver(model, input_orig, num_steps=args.fisher_steps, step_size=args.fisher_stepsize)
        # ensure the gradient is zero
        model.zero_grad()

        # compute output
        output_orig = model(input_orig)
        output_fisher = model(input_fisher)

        loss = criterion(output_orig, target)
        losses.update(loss.item(), input_orig.size(0))
        loss_fisher = kl_divergence(output_orig, output_fisher)
        losses_fisher.update(loss_fisher.item(), input_fisher.size(0))
        loss = loss + args.mult_coef*loss_fisher

        if torch.isnan(loss):
            print('nan appeared')
            import sys
            sys.exit(0)

        # measure accuracy and record loss
        prec = accuracy(output_orig, target)[0]
        top1.update(prec.item(), input_orig.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Loss Fisher {loss_fisher.val:.4f} ({loss_fisher.avg:.4f})\t'
                'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                epoch, i, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_fisher=losses_fisher, top1=top1))

def validate_normal_and_attack(val_loader, attack_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_attack = AverageMeter()
    top1 = AverageMeter()
    top1_attack = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec normal {top1.avg:.3f}% '.format(top1=top1))

    end = time.time()
    for i, (input, target) in enumerate(attack_loader):
        input, target = input.cuda(), target.cuda()
        input,target = pgd_attack(model, input, target, num_steps=20, step_size=2.0/255, random_start=False)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses_attack.update(loss.item(), input.size(0))
        top1_attack.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec attack {top1_attack.avg:.3f}% '.format(top1_attack=top1_attack))
    return top1.avg, top1_attack.avg

def save_checkpoint(state, fdir, fname):
    filepath = os.path.join(fdir, fname)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = args.lr
    elif epoch < 120:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=='__main__':
    main()
