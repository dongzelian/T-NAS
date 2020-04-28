import time

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from learner import Network
import utils.utils as utils
from copy import deepcopy

import pdb


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, criterion):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self._args = args
        self.update_lr_theta = args.update_lr_theta
        self.meta_lr_theta = args.meta_lr_theta
        self.update_lr_w = args.update_lr_w
        self.meta_lr_w = args.meta_lr_w
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.meta_batch_size = args.meta_batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.criterion = criterion
        self.model = Network(args, args.init_channels, args.n_way, args.layers, criterion).cuda()

        self.meta_optimizer_theta = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.meta_lr_theta, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.inner_optimizer_theta = torch.optim.SGD(self.model.arch_parameters(), lr=args.update_lr_theta)

        self.meta_optimizer_w = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr_w)
        self.inner_optimizer_w = torch.optim.SGD(self.model.parameters(), lr=self.update_lr_w)



    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def _update_theta_w_together(self, x_spt, y_spt, x_qry, y_qry):
        meta_batch_size, setsz, c_, h, w = x_spt.shape
        query_size = x_qry.shape[1]

        corrects_w = [0 for _ in range(self.update_step + 1)]
        corrects_theta = [0 for _ in range(self.update_step + 1)]

        ''' copy weight and gradient '''
        theta_clone = [v.clone() for v in self.model.arch_parameters()]
        for p in self.model.arch_parameters():
            p.grad = torch.zeros_like(p.data)
        theta_grad_clone = [p.grad.clone() for p in self.model.arch_parameters()]
        w_clone = dict([(k, v.clone()) for k, v in self.model.named_parameters()])
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)
        w_grad_clone = [p.grad.clone() for p in self.model.parameters()]

        for i in range(meta_batch_size):
            ''' Update w '''
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters())

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects_w[0] = corrects_w[0] + correct


            logits = self.model(x_spt[i], alphas=self.model.arch_parameters())  # x_spt.shape
            loss = self.criterion(logits, y_spt[i])
            self.inner_optimizer_w.zero_grad()
            self.inner_optimizer_theta.zero_grad()
            loss.backward()
            self.inner_optimizer_w.step()
            self.inner_optimizer_theta.step()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters())
                loss_q = self.criterion(logits_q, y_qry[i])
                # losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects_w[1] = corrects_w[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.model(x_spt[i], alphas=self.model.arch_parameters())
                loss = self.criterion(logits, y_spt[i])

                self.inner_optimizer_w.zero_grad()
                self.inner_optimizer_theta.zero_grad()
                loss.backward()
                self.inner_optimizer_w.step()
                self.inner_optimizer_theta.step()

                logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters())
                loss_q = self.criterion(logits_q, y_qry[i])

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects_w[k + 1] = corrects_w[k + 1] + correct

            ''' Use first-order gradient average '''
            self.inner_optimizer_w.zero_grad()
            self.inner_optimizer_theta.zero_grad()
            loss_q.backward()
            theta_grad_clone = [k + v.grad.clone() for k, v in zip(theta_grad_clone, self.model.arch_parameters())]
            w_grad_clone = [k + v.grad.clone() for k, v in zip(w_grad_clone, self.model.parameters())]
            for k, v in self.model.named_parameters():
                v.data.copy_(w_clone[k])
            for i in range(len(self.model.arch_parameters())):
               self.model.arch_parameters()[i].data.copy_(theta_clone[i])


        self.meta_optimizer_w.zero_grad()
        for k, v in zip(w_grad_clone, self.model.parameters()):
            v.grad.copy_(k / meta_batch_size)
        self.meta_optimizer_w.step()

        self.meta_optimizer_theta.zero_grad()
        for k, v in zip(theta_grad_clone, self.model.arch_parameters()):
            v.grad.copy_(k / meta_batch_size)
        self.meta_optimizer_theta.step()

        accs_w = np.array(corrects_w) / (query_size * meta_batch_size)

        return accs_w


    def forward(self, x_spt, y_spt, x_qry, y_qry, update_theta_w_time):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, query_size, c_, h, w]num_filter
        :param y_qry:   [b, query_size]
        :return:
        """
        start = time.time()
        accs= self._update_theta_w_together(x_spt, y_spt, x_qry, y_qry)
        update_theta_w_time.update(time.time() - start)

        return accs, update_theta_w_time


    def _update_theta_w_together_finetunning(self, model, inner_optimizer_theta, inner_optimizer_w, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4

        query_size = x_qry.shape[0]

        corrects_w = [0 for _ in range(self.update_step_test + 1)]
        corrects_theta = [0 for _ in range(self.update_step_test + 1)]

        with torch.no_grad():
            # [setsz, nway]
            logits_q = model(x_qry, alphas=model.arch_parameters())
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects_w[0] = corrects_w[0] + correct


        for k in range(self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = model(x_spt, alphas=model.arch_parameters())
            loss = self.criterion(logits, y_spt)

            inner_optimizer_w.zero_grad()
            inner_optimizer_theta.zero_grad()
            loss.backward()
            inner_optimizer_w.step()
            inner_optimizer_theta.step()

            with torch.no_grad():
                logits_q = model(x_qry, alphas=model.arch_parameters())
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects_w[k + 1] = corrects_w[k + 1] + correct


        accs_w = np.array(corrects_w) / query_size

        return accs_w


    def finetunning(self, x_spt, y_spt, x_qry, y_qry, update_theta_w_time, logging):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [query_size, c_, h, w]
        :param y_qry:   [query_size]
        :return:
        """
        #start = time.time()

        model = deepcopy(self.model)
        inner_optimizer_theta = torch.optim.SGD(model.arch_parameters(), lr=self.update_lr_theta)
        inner_optimizer_w = torch.optim.SGD(model.parameters(), lr=self.update_lr_w)

        start = time.time()
        accs_finetunning = self._update_theta_w_together_finetunning(model, inner_optimizer_theta,
                                                                                       inner_optimizer_w, x_spt, y_spt,
                                                                                       x_qry, y_qry)


        update_theta_w_time.update(time.time() - start)

        del model

        return accs_finetunning, update_theta_w_time



def main():
    pass


if __name__ == '__main__':
    main()
