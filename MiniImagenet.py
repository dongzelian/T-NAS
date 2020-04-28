import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batch_size, n_way, k_shot, k_query, resize, split=1, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batch_size: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batch_size = batch_size  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.split = split
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batch_size, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')  # image path

        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path

        self.data = []  # list of list
        self.img2label = {}  # img2label dict
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img601, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)  # 64

        ''' split train data '''
        if self.split != 1:
            data = list(zip(*self.data))
            data_list = [x for data_tuple in data for x in data_tuple]
            label_list = list(range(self.cls_num)) * len(data)
            self.data_target = list(zip(data_list, label_list))
            data_split = self.data_target[int(self.split[0] * len(data_list)):int(self.split[1] * len(data_list))]
            data_dict = {}
            for v in data_split:
                d = v[0]
                lbl = v[1]
                if lbl in data_dict.keys():
                    data_dict[lbl].append(d)
                else:
                    data_dict[lbl] = [d]

            self.data = []
            for i in range(self.cls_num):
                self.data.append(data_dict[i]) # list of list


        self.create_batch(self.batch_size)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]

        return dictLabels

    def create_batch(self, batch_size):
        """
        create batch for meta-learning.
        episode here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batch_size):  # for each batch
            # 1.select n_way classes randomly, choose 5-way from 64 classes, no duplicate
            selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)

            np.random.shuffle(selected_cls)
            support_x = []  # list of list
            query_x = []       # list of list
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class

                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, replace=False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)  # 5*1
            random.shuffle(query_x)    # 5*15

            self.support_x_batch.append(support_x)  # append set to current sets, 10000*5*1
            self.query_x_batch.append(query_x)  # append sets to current sets, 10000*5*15


    def __getitem__(self, index):
        support_x = torch.FloatTensor(self.set_size, 3, self.resize, self.resize)
        query_x = torch.FloatTensor(self.query_size, 3, self.resize, self.resize)


        flatten_support_x = []  # list of path
        for sublist in self.support_x_batch[index]:
            for item in sublist:
                flatten_support_x.append(os.path.join(self.path, item))


        support_y = []
        for sublist in self.support_x_batch[index]:
            for item in sublist:
                support_y.append(self.img2label[item[:9]])
        support_y = np.array(support_y)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [set_size]
        # query_y: [query_size]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way

        support_y_relative = np.zeros(self.set_size)  # 5*1
        query_y_relative = np.zeros(self.query_size)  # 5*15
        for idx, label in enumerate(unique):
            support_y_relative[support_y == label] = idx
            query_y_relative[query_y == label] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batch_size of sets, you can sample some small batch size of sets.
        return self.batch_size


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batch_size=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()
