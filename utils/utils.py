import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import itertools
from genotypes import PRIMITIVES, LSTM_PRIMITIVES, Genotype
from graphviz import Digraph
from collections import defaultdict

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
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


def imagewise_accuracy(output, target, pid):

    res = {}
    r = zip(output, target, pid)
    for o, t, p in r:
        tokens = p.split('_')
        organ = tokens[0]
        prob = o[t]
        if organ in res:
            res[organ].append(float(prob>=0.5))
        else:
            res[organ] = [float(prob>=0.5)]

    result = {}
    all = 0
    n = 0
    for k, v in res.items():
        s = np.sum(v)
        m = np.mean(v)
        result[k] = m
        all += s
        n += len(v)
    all_mean = all / n
    result['all'] = all_mean
    return result


def subjectiwise_accuracy(output, target, pid):

    res = {}
    r = zip(output, target, pid)
    for key, value in itertools.groupby(r, key=lambda x:x[-1]):
        tokens = key.split('_')
        organ = tokens[0]
        patentID = tokens[1]
        prob = [p[l] for p, l, id in value]
        max_p = max(prob)
        min_p = min(prob)
        mean_p = sum(prob)/len(prob)
        if organ in res:
            res[organ].append([float(max_p>=0.5), float(min_p>=0.5), float(mean_p>=0.5)])
        else:
            res[organ] = [[float(max_p>=0.5), float(min_p>=0.5), float(mean_p>=0.5)]]

    result = {}
    all = np.zeros((3,))
    n = 0
    for k, v in res.items():
        v = np.array(v)
        s = np.sum(v, axis=0)
        m = np.mean(v, axis=0)
        result[k] = list(m)
        all += s
        n += len(v)
    all_mean = all / n
    result['all'] = list(all_mean)
    return result






class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_mura(args):
    MURA_MEAN = [0.1524366]
    MURA_STD = [0.1807950]

    train_transform = transforms.Compose([
        transforms.RandomCrop(512, padding=args.padding),
        # transforms.ColorJitter(),
        transforms.RandomRotation(args.rotation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MURA_MEAN, MURA_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MURA_MEAN, MURA_STD),
    ])
    return train_transform, valid_transform

def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    #return np.sum(np.prod(v.size()) for v in model.model_parameters()) / 1e6
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x




# Written by Yin Zheng
def draw_genotype(genotype, n_nodes, filename):
    """
    :param genotype:
    :param filename:
    :return:
    """
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    steps = n_nodes

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for op, source, target in genotype:
        if source == 0:
            u = "c_{k-2}"
        elif source == 1:
            u = "c_{k-1}"
        else:
            u = str(source - 2)
        v = str(target-2)
        g.edge(u, v, label=op, fillcolor="gray")


    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=False)


def arch_to_genotype(arch_normal, arch_reduce, n_nodes, cell_type):
    if cell_type == "SAMPLE":
        primitives = PRIMITIVES
    elif cell_type == "LSTM" or cell_type == 'ENAS' or cell_type == "LSTM2":
        primitives = LSTM_PRIMITIVES
    else:
        assert False, "unsupported controller_type: %s" % cell_type

    gene_normal = [(primitives[op], f, t) for op, f, t in arch_normal]
    gene_reduce = [(primitives[op], f, t) for op, f, t in arch_reduce]
    concat = range(2, 2 + n_nodes)
    genotype = Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)
    return genotype


def infinite_get(data_iter, data_queue):
    try:
        data = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(data_queue)
        data = next(data_iter)
    return data, data_iter


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out


def arch_to_string(arch):
    return ', '.join(["(op:%d,from:%d,to:%d)" % (o, f, t) for o, f, t in arch])

def get_index_item(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs = int(inputs.item())
    return inputs
