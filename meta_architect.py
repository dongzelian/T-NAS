import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import pdb

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.update_step = args.update_step
        self.meta_optimizer_theta = torch.optim.Adam(self.model.arch_parameters(),
                                        lr=args.meta_lr_theta, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.inner_optimizer_w = torch.optim.SGD(self.model.parameters(), lr=args.update_lr_w)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def _update_theta(self, x_spt, y_spt, x_qry, y_qry, criterion):
        meta_batch_size, setsz, c_, h, w = x_spt.shape
        #query_size = x_qry.shape[1]

        #losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        #corrects = [0 for _ in range(self.update_step + 1)]

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

    def step(self, x_spt_search, y_spt_search, x_qry_search, y_qry_search, criterion):
        self._update_theta(x_spt_search, y_spt_search, x_qry_search, y_qry_search, criterion)

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

