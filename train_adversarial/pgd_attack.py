from __future__ import print_function

import torch
import numpy as np

class LinfPGDAttack:
    def __init__(self, model, num_steps, step_size, random_start, loss_func='xent'):
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        if loss_func == 'xent':
            self.loss_func = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        elif loss_func == 'cw':
            raise NotImplementedError('I am regret to inform you that no cw yet')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = torch.nn.CrossEntropyLoss().cuda()

    def perturb(self, x_nat, y, epsilon):
        if self.rand:
            x = x_nat + (2*epsilon*torch.rand(x_nat.shape) - epsilon).cuda()
            x = torch.clamp(x, 0.0, 1.0)
        else:
            x = x_nat

        x_adv = x.clone().requires_grad_(True)

        for i in range(self.num_steps):
            pred = self.model(x_adv)
            cost = -self.loss_func(pred, y).mean()
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()
            x_adv.grad.sign_()
            x_adv.detach().sub_(self.step_size*x_adv.grad)
            diff = x_adv - x_nat
            diff.clamp_(-epsilon, epsilon)
            x_adv.detach().copy_((diff + x_nat).clamp_(0.0, 1.0))

        self.model.zero_grad()
        return x_adv

class LinfFisherSolver:
    def __init__(self, model, num_steps, step_size):
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size

    def _kl_divergence(self, P, Q):
        eps = 1e-8
        softmax = torch.nn.Softmax(dim=1)
        P_prob = softmax(P)
        Q_prob = softmax(Q)
        KL = P_prob*torch.log((P_prob+eps)/(Q_prob+eps))
        KL = KL.sum(dim=1).mean()
        return KL

    def solve(self, x_nat, epsilon):
        x = x_nat + (2*epsilon*torch.rand(x_nat.shape) - epsilon).cuda()
        x = torch.clamp(x, 0.0, 1.0)

        x_fisher = x.clone().requires_grad_(True)

        pred_p = self.model(x).detach()
        for i in range(self.num_steps):
            pred_q = self.model(x_fisher)
            cost = -self._kl_divergence(pred_p, pred_q)

            self.model.zero_grad()
            if x_fisher.grad is not None:
                x_fisher.grad.data.fill_(0)
            cost.backward()
            x_fisher.grad.sign_()
            x_fisher.detach().sub_(self.step_size*x_fisher.grad)
            # projected
            diff = x_fisher - x_nat
            diff.clamp_(-epsilon, epsilon)
            x_fisher.detach().copy_((diff + x_nat).clamp_(0.0, 1.0))

        self.model.zero_grad()
        return x_fisher
