import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import h5py
import json
import argparse
import os

import shutil
import torch.nn.functional as F
import pdb

class SimpleHDF5Dataset:
    def __init__(self, file_handle):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats']
        self.all_labels = self.f['all_labels']
        self.total = self.f['count']

    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

# a dataset to allow for category-uniform sampling of base and novel classes.
# also incorporates hallucination
class LowShotDataset:
    def __init__(self, base_feats, novel_feats, base_classes, novel_classes):
        self.f = base_feats
        self.all_base_feats_dset = self.f['all_feats'][...]
        self.all_base_labels_dset = self.f['all_labels'][...]

        self.novel_feats = novel_feats['all_feats']
        self.novel_labels = novel_feats['all_labels']

        self.base_classes = base_classes
        self.novel_classes = novel_classes

        self.frac = 0.5
        self.all_classes = np.concatenate((base_classes, novel_classes))

    def sample_base_class_examples(self, num):
        sampled_idx = np.sort(np.random.choice(len(self.all_base_labels_dset), num, replace=False))
        return torch.Tensor(self.all_base_feats_dset[sampled_idx,:]), torch.LongTensor(self.all_base_labels_dset[sampled_idx].astype(int))

    def sample_novel_class_examples(self, num):
        sampled_idx = np.random.choice(len(novel_feats['all_labels']), num)
        return torch.Tensor(self.novel_feats[sampled_idx,:]), torch.LongTensor(self.novel_labels[sampled_idx].astype(int))

    def get_sample(self, batchsize):
        num_base = round(self.frac*batchsize)
        num_novel = batchsize - num_base
        base_feats, base_labels = self.sample_base_class_examples(int(num_base))
        novel_feats, novel_labels = self.sample_novel_class_examples(int(num_novel))
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def get_index_base_novel_sample(self, batchsize, cur_novel_l, related_index):
        novel_feats, novel_labels = self.sample_novel_class_examples(int(batchsize))
        base_sample_index = []
        for curent_label in novel_labels:
            idx = np.where(cur_novel_l == curent_label.data.numpy())[0][0]
            idy = related_index.data.numpy()[idx]
            idxx = np.random.choice(idy, 1)[0]
            idyy = np.random.choice(np.where(self.all_base_labels_dset==idxx)[0], 1)[0]
            base_sample_index.append(idyy)

        base_feats, base_labels = torch.Tensor(self.all_base_feats_dset[base_sample_index,:]), torch.LongTensor(self.all_base_labels_dset[base_sample_index].astype(int))
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def featdim(self):
        return self.novel_feats.shape[1]

# simple data loader for test
def get_test_loader(file_handle, batch_size=1000):
    testset = SimpleHDF5Dataset(file_handle)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return data_loader

def construct_patchmix(x, rotated_x, chunks = 5, alpha=0.1):
    # row_x = torch.chunk(x, chunks, dim = -1)
    # col_x = [torch.chunk(row_x_, chunks, dim = -2) for row_x_ in row_x]    

    # row_rotated_x = torch.chunk(rotated_x, chunks, dim = -1)
    # col_rotated_x = [torch.chunk(row_x_, chunks, dim = -2) for row_x_ in row_rotated_x] 


    therold = therold = torch.rand(512)
    # pdb.set_trace()
    therold[therold >= alpha] = 1
    therold[therold < alpha] = 0

    new_x = therold.cuda() * x + (1 - therold.cuda()) * rotated_x
    # pdb.set_trace()

    # new_col_x = [therold[i][j] * col_x[i][j] + (1 - therold[i][j]) * col_rotated_x[i][j] for i in range(chunks) for j in range(chunks)]

    # cut_list = [new_col_x[i*chunks: (i+1)*chunks] for i in range(chunks)]
    # new_x = torch.cat([torch.cat(cut_list[i], dim = -2) for i in range(chunks)], dim = -1)

    return new_x

def CrossEntropy(pred, target, scale = False):
    pred = pred.softmax(-1)
    loss = -torch.log(pred) * target
    if scale:
        loss = loss.sum() / ((target > 0).sum() + 0.000001)
    else:
        loss = loss.sum() / (target.sum() + 0.000001)

    return loss

def BinaryEntropy(pred, target, scale = False):
    pred = pred.sigmoid()
    loss = -torch.log(pred + 0.0000001) * target
    if scale:
        loss = loss.sum() / ((target > 0).sum() + 0.000001)
    else:
        loss = loss.sum() / (target.sum() + 0.000001)
    return loss

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def training_loop(lowshot_dataset,novel_test_feats, num_classes, params, batchsize=1000, maxiters=1000, nTimes = 0):
    if os.path.exists('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '/') == False:
        os.makedirs('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '/')
    if os.path.exists('Model_SHOT5/' + params.name + '/' + str(nTimes) + '_' + str(params.alpha) + '_' + str(params.beta)) == False:
        os.makedirs('Model_SHOT5/' + params.name + '/' + str(nTimes) + '_' + str(params.alpha) + '_' + str(params.beta))

    featdim = 512
    model = torch.nn.utils.weight_norm(nn.Linear(featdim, num_classes), dim=0)
    # model = nn.Linear(featdim, num_classes)
    model = model.cuda()
    
    test_loader = get_test_loader(novel_test_feats)
    
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()

    best_ACC = 0.0
    tmp_epoach = 0
    tmp_count = 0
    tmp_rate = params.lr
    recode_reload = {}
    reload_model = False
    max_tmp_count = 10
    optimizer = torch.optim.Adam(model.parameters(), tmp_rate, weight_decay=params.wd)
    t_embedding_100 = torch.FloatTensor(np.load('MiniImageNetWord2Vec.npy'))
    t_original_relation = F.normalize(t_embedding_100,dim=-1).mm(F.normalize(t_embedding_100,dim=-1).t())

    novel_labels = list(set(lowshot_dataset.novel_labels))
    t_original_relation = t_original_relation[novel_labels][:,:64]
    _, index = torch.topk(t_original_relation, params.beta, dim=-1)
    
    for epoch in range(maxiters):

        optimizer.zero_grad()
        (x,y) = lowshot_dataset.get_index_base_novel_sample(batchsize, novel_labels, index)
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        x_base, x_novel = torch.chunk(x, 2, dim = 0)
        y_base, y_novel = torch.chunk(y, 2, dim = 0)

        # x_base = x_base.view(x_base.shape[0], x_base.shape[1], -1).mean(dim=2)
        # x_novel = x_novel.view(x_novel.shape[0], x_novel.shape[1], -1).mean(dim=2)

        lam = np.random.beta(2.0, 2.0)
        mixup_feat = construct_patchmix(lam * x_novel, (1 - lam) * x_base, alpha=params.alpha) 

        # novel_feat_x = x_novel.view(x_novel.shape[0], x_novel.shape[1], -1).mean(dim=2)
        # mixup_feat = new_x.view(new_x.shape[0], new_x.shape[1], -1).mean(dim=2)

        all_feats = torch.cat((x_novel, mixup_feat),0) 

        scores_novel, scores_mixup = torch.chunk(model(all_feats), 2, dim = 0)

        loss_1 = F.cross_entropy(scores_novel, y_novel)
        loss_3 = F.cross_entropy(scores_mixup, y_novel) * lam

        loss = loss_1  + loss_3

        loss.backward()
        optimizer.step()

    return model

def perelement_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def eval_loop(data_loader, model, base_classes, novel_classes):
    model = model.eval()
    top1 = None
    top5 = None
    no_novel_class = list(set(range(100)).difference(set(novel_classes)))
    all_labels = None
    for i, (x,y) in enumerate(data_loader):
        x = Variable(x.cuda())
        # feat_x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        feat_x = x
        scores = model(feat_x).softmax(-1)
        # pdb.set_trace()
        scores[:,no_novel_class] = -0.0
        top1_this, _ = perelement_accuracy(scores.data, y)
        top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
        all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

    is_novel = np.in1d(all_labels, novel_classes)
    top1_novel = np.mean(top1[is_novel])
    return [top1_novel]


def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--name', default='random_mask_512_Semantic', type=str)
    parser.add_argument('--numclasses', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--maxiters', default=2002, type=int)
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=5, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    with open('ExperimentSplit/Json/base_classes_train_meta.json','r') as f:
        exp = json.load(f)
        base_classes = list(set(exp['image_labels']))

    with open('ExperimentSplit/Json/base_classes_val_meta.json','r') as f:
        exp = json.load(f)
        novel_classes = list(set(exp['image_labels']))

    for feature_path in os.listdir('Features/'):
        if '512' not in feature_path:
            continue
        
        train_feats = h5py.File('Features/' + feature_path + '/train.hdf5', 'r')
        test_feats = h5py.File('Features/' + feature_path + '/val.hdf5', 'r')
        all_feats_dset = test_feats['all_feats'][...]
        all_labels = test_feats['all_labels'][...]

        start_ = 0
        end_ = 600

        n_shot = 5
        for alpha in [0.5, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.1]:
            start_ = 0
            end_ = 600 
            params.alpha = alpha
            if os.path.exists('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '/'):
                len_results = len(os.listdir('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.alpha) + '_' + str(params.beta) + '/'))
                if len_results >= 600:
                    continue
                else:
                   start_ = len_results

            lowshot_dataset = None
            for nTime in range(start_, end_):
                # print(nTime)

                selected = np.random.choice(novel_classes, 5, replace=False)

                novel_train_feats = []
                novel_train_labels = []
                novel_test_feats = []
                novel_test_labels = []

                for K in selected:
                    is_K = np.in1d(all_labels, K)

                    current_idx = np.random.choice(np.sum(is_K), 15 + n_shot, replace=False)
                    novel_train_feats.append(all_feats_dset[is_K][current_idx[:n_shot]])
                    novel_test_feats.append(all_feats_dset[is_K][current_idx[n_shot:]])

                    for _ in range(n_shot):
                        novel_train_labels.append(K)
                    for _ in range(15):
                        novel_test_labels.append(K)

                novel_train_feats  =  np.vstack(novel_train_feats)
                novel_train_labels =  np.array(novel_train_labels)
                novel_test_feats   =  np.vstack(novel_test_feats)
                novel_test_labels  =  np.array(novel_test_labels)

                novel_feats = {}
                novel_feats['all_feats'] = novel_train_feats
                novel_feats['all_labels'] = novel_train_labels
                novel_feats['count'] = len(novel_train_labels)

                novel_val_feats = {}
                novel_val_feats['all_feats'] = novel_test_feats
                novel_val_feats['all_labels'] = novel_test_labels
                novel_val_feats['count'] = len(novel_test_labels)

                if lowshot_dataset is not None:
                    lowshot_dataset.novel_feats = novel_feats['all_feats']
                    lowshot_dataset.novel_labels = novel_feats['all_labels']
                    lowshot_dataset.novel_classes = novel_classes
                    lowshot_dataset.all_classes = np.concatenate((base_classes, novel_classes))
                else:
                    lowshot_dataset = LowShotDataset(train_feats, novel_feats, base_classes, novel_classes)

                model = training_loop(lowshot_dataset, novel_val_feats, params.numclasses, params, params.batchsize, params.maxiters, nTimes = nTime)

                # print('trained')

