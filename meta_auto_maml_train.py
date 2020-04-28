import time

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import genotypes

from model import NetworkMiniImageNet
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

        self.update_lr_theta = args.update_lr_theta
        self.meta_lr_theta = args.meta_lr_theta
        self.update_lr_w = args.update_lr_w
        self.meta_lr_w = args.meta_lr_w
        self.weight_decay = args.weight_decay
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.meta_batch_size = args.meta_batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.criterion = criterion
        genotype = eval("genotypes.%s" % args.arch)
        auxiliary = None
        self.model = NetworkMiniImageNet(args, args.init_channels, args.n_way, args.layers, criterion, auxiliary, genotype)


        self.meta_optimizer_w = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr_w, weight_decay=self.weight_decay)
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

        return total_norm / counter


    def _update_w(self, x_spt, y_spt, x_qry, y_qry):
        meta_batch_size, setsz, c_, h, w = x_spt.shape
        query_size = x_qry.shape[1]

        corrects = [0 for _ in range(self.update_step + 1)]

        ''' copy weight and gradient '''
        w_clone = dict([(k, v.clone()) for k, v in self.model.named_parameters()])
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)
        grad_clone = [p.grad.clone() for p in self.model.parameters()]

        for i in range(meta_batch_size):

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i])

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # 1. run the i-th task and compute loss for k=0

            logits = self.model(x_spt[i])  # x_spt.shape
            loss = self.criterion(logits, y_spt[i])
            self.inner_optimizer_w.zero_grad()
            loss.backward()
            self.inner_optimizer_w.step()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.model(x_spt[i])
                loss = self.criterion(logits, y_spt[i])

                self.inner_optimizer_w.zero_grad()
                loss.backward()
                self.inner_optimizer_w.step()

                logits_q = self.model(x_qry[i])
                loss_q = self.criterion(logits_q, y_qry[i])

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

            ''' Use first-order gradient average '''
            self.inner_optimizer_w.zero_grad()
            loss_q.backward()
            grad_clone = [k + v.grad.clone() for k, v in zip(grad_clone, self.model.parameters())]
            for k, v in self.model.named_parameters():
                v.data.copy_(w_clone[k])

        self.meta_optimizer_w.zero_grad()
        for k, v in zip(grad_clone, self.model.parameters()):
            v.grad.copy_(k / meta_batch_size)

        self.meta_optimizer_w.step()
        accs = np.array(corrects) / (query_size * meta_batch_size)

        return accs

    def forward(self, x_spt, y_spt, x_qry, y_qry, update_w_time):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, query_size, c_, h, w]num_filter
        :param y_qry:   [b, query_size]
        :return:
        """
        start = time.time()
        accs_w = self._update_w(x_spt, y_spt, x_qry, y_qry)
        update_w_time.update(time.time() - start)

        return accs_w, update_w_time


    def _update_w_finetunning(self, model, inner_optimizer_w, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4

        query_size = x_qry.shape[0]

        corrects = [0 for _ in range(self.update_step_test + 1)]


        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = model(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # 1. run the i-th task and compute loss for k=0
        logits = model(x_spt)
        loss = self.criterion(logits, y_spt)
        inner_optimizer_w.zero_grad()
        loss.backward()
        inner_optimizer_w.step()

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = model(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):

            logits = model(x_spt)
            loss = self.criterion(logits, y_spt)

            inner_optimizer_w.zero_grad()
            loss.backward()
            inner_optimizer_w.step()

            logits_q = model(x_qry)

            loss_q = self.criterion(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        accs = np.array(corrects) / query_size

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, update_w_time):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [query_size, c_, h, w]
        :param y_qry:   [query_size]
        :return:
        """
        model = deepcopy(self.model)
        inner_optimizer_w = torch.optim.SGD(model.parameters(), lr=self.update_lr_w)
        start = time.time()
        accs_w_finetunning = self._update_w_finetunning(model, inner_optimizer_w, x_spt, y_spt, x_qry, y_qry)
        update_w_time.update(time.time() - start)

        del model

        return accs_w_finetunning, update_w_time


def main():
    pass


if __name__ == '__main__':
    main()
