import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
sys.path.insert(0, '..')

from utils import *
from lenet import *

parser = argparse.ArgumentParser()
parser.add_argument('--state-dict', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--attack-algo', default='pgd', type=str, metavar='PATH', help='algorithm to attack')
parser.add_argument('--num-steps', type=int, default=40)
parser.add_argument('--step-size', type=float, default=0.01)
parser.add_argument('--linear', dest='linear', action='store_true')
args = parser.parse_args()

if __name__=="__main__":
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # model
    print("if linear")
    print(args.linear)
    if args.linear:
        model = NetLinear().cuda()
    else:
        model = Net().cuda()
    # state dict
    saved_state_dict = torch.load(args.state_dict)
    model.load_state_dict(saved_state_dict)

    print(model)
    print(args.state_dict)

    model.eval()
    if args.attack_algo == 'deepfool':
        p_adv = deepfool_attack(model, test_loader, device)
        print('p_adv: %f'%p_adv)
    elif args.attack_algo == 'fgsm':
        #fsgm_attack(model, test_loader, device)
        raise NotImplementedError('deprecated, use PGD Attack instead')
    elif args.attack_algo == 'pgd':
        pgd_attack_mnist_test(model, device, args.num_steps, args.step_size)
    else:
        raise NotImplementedError('wrong option')
