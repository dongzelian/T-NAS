import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype

import pdb

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, args, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, pretrained=False):
        super(Network, self).__init__()
        self._args = args
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C



        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C_curr // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr // 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_curr // 2, C_curr, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.MaxPool2d(2, 2),
        )



        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False


        if layers == 2:
            for i in range(layers):
                if i == 1:
                    C_curr *= 2
                    reduction = True
                else:
                    reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                reduction_prev = reduction
                self.cells += [cell]
                C_prev_prev, C_prev = C_prev, multiplier*C_curr


        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        if self._args.checkname == 'darts' or self._args.checkname == 'enas':
            self.classifier = nn.Linear(C_prev, num_classes)
        else:
            self.classifier_meta_nas = nn.Linear(C_prev, num_classes)
        if pretrained == True:
            self._load_pretrained_alphas()
        else:
            self._initialize_alphas()



    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, alphas):
        s0 = s1 = self.stem0(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self._args.layers == 1:
                    weights = F.softmax(alphas[0], dim=-1)
                else:
                    weights = F.softmax(alphas[1], dim=-1)
            else:
                weights = F.softmax(alphas[0], dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        if self._args.checkname == 'darts' or self._args.checkname == 'enas':
            logits = self.classifier(out.view(out.size(0), -1))
        else:
            logits = self.classifier_meta_nas(out.view(out.size(0), -1))

        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = 1e-3*nn.init.normal_(torch.empty(k, num_ops)).cuda()   # self.alphas_normal.shape = [14, 8]
        self.alphas_normal.requires_grad = True
        self.alphas_reduce = 1e-3*nn.init.normal_(torch.empty(k, num_ops)).cuda()
        self.alphas_reduce.requires_grad = True
        if self._args.layers == 1:
            self._arch_parameters = [
                self.alphas_reduce
            ]
        else:
            self._arch_parameters = [
                self.alphas_normal,
                self.alphas_reduce,
            ]

    def _load_pretrained_alphas(self):
        self.alphas_normal = torch.load(self._args.pretrained_model)['state_dict_theta'][0].cuda()   # self.alphas_normal.shape = [14, 8]
        self.alphas_normal.requires_grad = True
        self.alphas_reduce = torch.load(self._args.pretrained_model)['state_dict_theta'][1].cuda()
        self.alphas_reduce.requires_grad = True
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n

                W = weights[start:end].copy()

                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        if self._args.layers == 1:
            gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
            concat = range(2+self._steps-self._multiplier, self._steps+2)
            genotype = Genotype(
                normal=[], normal_concat=[],
                reduce=gene_reduce, reduce_concat=concat
            )
        else:
            gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

            concat = range(2+self._steps-self._multiplier, self._steps+2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat
            )

        return genotype

