import time

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import genotypes

from learner import Network
from model import NetworkMiniImageNet, NetworkOmniglot
import utils.utils as utils
from darts_architect import Architect
from copy import deepcopy

import pdb


class Meta_decoding(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, criterion, genotype, pretrained=False):
        """

        :param args:
        """
        super(Meta_decoding, self).__init__()

        self.dataset = args.dataset
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

        auxiliary = None
        if self.dataset == 'Omniglot':
            self.model = NetworkOmniglot(args, args.init_channels, args.n_way, args.layers, criterion, auxiliary,
                                         genotype)
        else:
            self.model = NetworkMiniImageNet(args, args.init_channels, args.n_way, args.layers, criterion, auxiliary,
                                             genotype)

        self.meta_optimizer_w = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr_w)
        self.inner_optimizer_w = torch.optim.SGD(self.model.parameters(), lr=self.update_lr_w)


        if pretrained == True:
            pretrain_dict = torch.load('/data2/dongzelian/NAS/meta_nas/run_meta_nas/mini-imagenet/meta-nas/experiment_18/model_best.pth.tar')['state_dict_w']
            model_dict = {}
            state_dict = self.model.state_dict()
            for k, v in pretrain_dict.items():
                if k[6:] in state_dict:
                    model_dict[k[6:]] = v
                else:
                    print(k)
            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)

            self.model._arch_parameters = torch.load('/data2/dongzelian/NAS/meta_nas/run_meta_nas/mini-imagenet/meta-nas/experiment_18/model_best.pth.tar')['state_dict_theta']


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

        #losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
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
                #loss_q = self.criterion(logits_q, y_qry[i])
                #losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # 1. run the i-th task and compute loss for k=0

            logits = self.model(x_spt[i])  # x_spt.shape
            loss = self.criterion(logits, y_spt[i])
            # grad = torch.autograd.grad(loss, self.model.parameters())
            # fast_weights = list(map(lambda p: p[1] - self.update_lr_w * p[0], zip(grad, self.model.parameters())))
            self.inner_optimizer_w.zero_grad()
            loss.backward()
            self.inner_optimizer_w.step()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i])
                #loss_q = self.criterion(logits_q, y_qry[i])
                #losses_q[1] += loss_q
                # [setsz]
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
                '''
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr_w * p[0], zip(grad, fast_weights)))
                '''
                logits_q = self.model(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.criterion(logits_q, y_qry[i])
                #losses_q[k + 1] += loss_q

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

        # end of all tasks
        # sum over all losses on query set across all tasks
        ''' when meta_batch_size == 1 '''
        # loss_q = losses_q[-1] / meta_batch_size
        # loss_q.backward()
        # pdb.set_trace()

        # print('meta update')
        # for p in self.model.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optimizer_w.step()

        accs = np.array(corrects) / (query_size * meta_batch_size)

        return accs








        ''' copy weight and gradient '''
        w_clone = dict([(k, v.clone()) for k, v in self.model.named_parameters()])
        for p in self.model.arch_parameters():
            p.grad = torch.zeros_like(p.data)
        grad_clone = [p.grad.clone() for p in self.model.arch_parameters()]

        for i in range(meta_batch_size):
            for k in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.model(x_spt[i], alphas=self.model.arch_parameters())
                loss = criterion(logits, y_spt[i])

                self.inner_optimizer_w.zero_grad()
                loss.backward()
                self.inner_optimizer_w.step()

            ''' Compute loss of final step '''
            logits_q = self.model(x_qry[i], alphas=self.model.arch_parameters())
            loss_q = criterion(logits_q, y_qry[i])
            ''' Use first-order gradient average '''
            self.inner_optimizer_w.zero_grad()
            # set 0 for theta
            self.meta_optimizer_theta.zero_grad()
            #pdb.set_trace()
            for k, v in self.model.named_parameters():
                v.data.copy_(w_clone[k])
            loss_q.backward()
            grad_clone = [k + v.grad.clone() for k, v in zip(grad_clone, self.model.arch_parameters())]


        # optimize theta parameters
        self.meta_optimizer_theta.zero_grad()
        for k, v in zip(grad_clone, self.model.arch_parameters()):
            v.grad.copy_(k / meta_batch_size)
        self.meta_optimizer_theta.step()



    def _update_w_efficient(self, x_spt, y_spt, x_qry, y_qry):
        meta_batch_size, setsz, c_, h, w = x_spt.shape
        query_size = x_qry.shape[1]

        #losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
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
                #loss_q = self.criterion(logits_q, y_qry[i])
                #losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # 1. run the i-th task and compute loss for k=0

            logits = self.model(x_spt[i])  # x_spt.shape
            loss = self.criterion(logits, y_spt[i])
            # grad = torch.autograd.grad(loss, self.model.parameters())
            # fast_weights = list(map(lambda p: p[1] - self.update_lr_w * p[0], zip(grad, self.model.parameters())))
            self.inner_optimizer_w.zero_grad()
            loss.backward()
            self.inner_optimizer_w.step()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i])
                #loss_q = self.criterion(logits_q, y_qry[i])
                #losses_q[1] += loss_q
                # [setsz]
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
                '''
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr_w * p[0], zip(grad, fast_weights)))
                '''
                logits_q = self.model(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.criterion(logits_q, y_qry[i])
                #losses_q[k + 1] += loss_q

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

        # end of all tasks
        # sum over all losses on query set across all tasks
        ''' when meta_batch_size == 1 '''
        # loss_q = losses_q[-1] / meta_batch_size
        # loss_q.backward()
        # pdb.set_trace()

        # print('meta update')
        # for p in self.model.parameters()[:5]:
        # 	print(torch.norm(p).item())
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

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.model

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = model(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # 1. run the i-th task and compute loss for k=0
        logits = model(x_spt)
        loss = self.criterion(logits, y_spt)
        inner_optimizer_w.zero_grad()
        loss.backward()
        inner_optimizer_w.step()
        # grad = torch.autograd.grad(loss, model.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr_w * p[0], zip(grad, model.parameters())))

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
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = model(x_spt)
            loss = self.criterion(logits, y_spt)

            inner_optimizer_w.zero_grad()
            loss.backward()
            inner_optimizer_w.step()
            '''
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr_w * p[0], zip(grad, fast_weights)))
            '''
            logits_q = model(x_qry)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            #loss_q = self.criterion(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        accs = np.array(corrects) / query_size

        return accs

    def _update_theta_w_together_finetunning(self, model, inner_optimizer_theta, inner_optimizer_w, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4

        for k in range(self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = model(x_spt, alphas=model.arch_parameters())
            loss = self.criterion(logits, y_spt)

            inner_optimizer_w.zero_grad()
            inner_optimizer_theta.zero_grad()
            loss.backward()
            inner_optimizer_w.step()
            inner_optimizer_theta.step()


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
