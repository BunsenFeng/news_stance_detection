import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing, RGCNConv, GATConv
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import math
import numpy as np


def get_metric(probs, labels):
    # hit 123
    hit1 = 0
    hit2 = 0
    hit3 = 0
    for i in range(len(labels)):
        temp = probs[i].clone()
        if torch.argmax(temp) == labels[i]:
            hit1 += 1
            hit2 += 1
            hit3 += 1
            continue
        temp[torch.argmax(temp)] = 0
        if torch.argmax(temp) == labels[i]:
            hit2 += 1
            hit3 += 1
            continue
        temp[torch.argmax(temp)] = 0
        if torch.argmax(temp) == labels[i]:
            hit3 += 1
            continue
    hit1 = hit1 / len(labels)
    hit2 = hit2 / len(labels)
    hit3 = hit3 / len(labels)
    # F1 and accs
    TP = [0,0,0,0,0]
    TN = [0,0,0,0,0]
    FP = [0,0,0,0,0]
    FN = [0,0,0,0,0]
    for i in range(len(labels)):
        temp = probs[i]
        if torch.argmax(temp) == labels[i]:
            TP[labels[i]] += 1
            for j in range(5):
                if not j == labels[i]:
                    TN[j] += 1
        else:
            FP[torch.argmax(temp)] += 1
            FN[labels[i]] += 1
            for j in range(5):
                if not j == torch.argmax(temp) and not j == labels[i]:
                    TN[j] += 1
    
    precision = [TP[i] / max(TP[i] + FP[i], 1) for i in range(5)]
    recall = [TP[i] / max(TP[i] + FN[i], 1) for i in range(5)]
    F1 = [2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1) for i in range(5)]
    #macro_precision = sum(precision) / 5
    #macro_recall = sum(recall) / 5
    macro_F1 = sum(F1) / 5
    micro_precision = sum(TP) / (sum(TP) + sum(FP))
    micro_recall = sum(TP) / (sum(TP) + sum(FN))
    assert (micro_precision == micro_recall)
    assert (micro_precision == hit1)
    micro_F1 = micro_precision
    return {'hit1':hit1, 'hit2':hit2, 'hit3':hit3, 'micro_F1':micro_F1, 'macro_F1':macro_F1}

def pad_collate(x):
    return x

class PSPDataset(Dataset):
    def __init__(self,  name, test_fold):  # name = train/dev
        path = '/new_temp/fsb/fsb_PolitiStance/'

        self.name = name
        self.idlist = []

        if self.name == 'train':
            for i in range(10):
                if i == test_fold:
                    continue
                f = open(path + 'fold/fold_' + str(i) + '.txt')
                for line in f:
                    self.idlist.append(int(line.strip()))
                f.close()
            self.length = len(self.idlist)
        if self.name == 'test':
            f = open(path + 'fold/fold_' + str(test_fold) + '.txt')
            for line in f:
                self.idlist.append(int(line.strip()))
            f.close()
            self.length = len(self.idlist)

        self.raw_truth = []
        f = open(path + 'ground_truth.txt')
        for line in f:
            if line.strip() == 'True':
                self.raw_truth.append(1)
            if line.strip() == 'False':
                self.raw_truth.append(0)

        self.ground_truth = []
        for id in self.idlist:
            self.ground_truth.append(self.raw_truth[id])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        temp = torch.load('graph/graph_' + str(self.idlist[index]) + '.pt')
        temp['ground_truth'] = self.ground_truth[index]
        #temp['roberta_feature'] = temp['roberta_feature'][1:]
        length = len(temp['roberta_feature'])
        temp['edge_index'] = temp['edge_index'][:,length+1:]
        temp['edge_type'] = temp['edge_type'][length+1:]
        return temp

# gated RGCN in a nutshell
# in_channels: text encoding dimension, out_channels: dim for each node rep, num_relations
# Input: node_features:torch.size([node_cnt,in_channels]), query_features = torch.size([in_channels]) (MISSING IN DATA FILE)
# edge_index = torch.size([[headlist],[taillist]]), edge_type = torch.size([typelist])
# Output: node representation of torch.size([node_cnt, out_channels])
class GatedRGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GatedRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.RGCN1 = RGCNConv(in_channels = out_channels, out_channels = out_channels, num_relations = num_relations)
        self.attention_layer = nn.Linear(2 * out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.attention_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, node_features, edge_index, edge_type):

        #layer 1
        #print(node_features.size())
        #print(edge_index.size())
        #print(edge_type.size())
        u_0 = self.RGCN1(node_features, edge_index, edge_type)
        a_1 = self.sigmoid(self.attention_layer(torch.cat((u_0, node_features),dim=1)))
        h_1 = self.tanh(u_0) * a_1 + node_features * (1 - a_1)

        return h_1

class PSPDetector(pl.LightningModule):
    def __init__(self, semantic_in_channels, entity_in_channels, out_channels, dropout, num_relations):
        super().__init__()
        self.semantic_in_channels = semantic_in_channels
        self.entitiy_in_channels = entity_in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.linear_before_RGCN_semantic = nn.Linear(self.semantic_in_channels, self.out_channels)
        self.linear_before_RGCN_entity = nn.Linear(self.entitiy_in_channels, self.out_channels)
        self.GatedRGCN = GatedRGCN(self.out_channels, self.out_channels, self.num_relations)
        self.linear_classification = nn.Linear(self.out_channels, 2)

        torch.nn.init.kaiming_uniform(self.linear_before_RGCN_semantic.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform(self.linear_before_RGCN_entity.weight, nonlinearity='leaky_relu')

        self.dropout_layer = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        #self.KLDivLoss = nn.KLDivLoss(size_average=False, reduction='sum')
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return x

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(),)
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5)
        #optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        avg_loss = 0
        acc = 0
        for graph in train_batch:
            semantic_feature = self.dropout_layer(self.relu(self.linear_before_RGCN_semantic(graph['roberta_feature'])))
            try:
                entity_feature = self.dropout_layer(self.relu(self.linear_before_RGCN_entity(graph['entity_feature'])))
                node_feature = torch.cat((semantic_feature, entity_feature))
            except:
                node_feature = semantic_feature
            node_feature = self.dropout_layer(self.relu(self.GatedRGCN(node_feature, graph['edge_index'], graph['edge_type'])))
            node_feature = self.dropout_layer(self.relu(self.GatedRGCN(node_feature, graph['edge_index'], graph['edge_type'])))
            length = len(graph['roberta_feature'])
            prob = self.linear_classification(torch.mean(node_feature[:len(graph['roberta_feature'])], dim = 0))
            loss = self.CELoss(prob.unsqueeze(0), torch.tensor(graph['ground_truth']).unsqueeze(0).long().cuda())
            avg_loss += loss
            if torch.argmax(prob) == graph['ground_truth']:
                acc += 1
        avg_loss /= len(train_batch)
        acc /= len(train_batch)

        self.log('train_loss', avg_loss.item())
        self.log('train_acc', acc)
        return avg_loss

    def validation_step(self, val_batch, batch_idx):
        #print(val_batch[0])

        avg_loss = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for graph in val_batch:
            semantic_feature = self.relu(self.linear_before_RGCN_semantic(graph['roberta_feature']))
            try:
                entity_feature = self.dropout_layer(self.relu(self.linear_before_RGCN_entity(graph['entity_feature'])))
                node_feature = torch.cat((semantic_feature, entity_feature))
            except:
                node_feature = semantic_feature

            node_feature = self.relu(self.GatedRGCN(node_feature, graph['edge_index'], graph['edge_type']))
            node_feature = self.relu(self.GatedRGCN(node_feature, graph['edge_index'], graph['edge_type']))
            length = len(graph['roberta_feature'])
            prob = self.linear_classification(torch.mean(node_feature[:len(graph['roberta_feature'])], dim = 0))
            loss = self.CELoss(prob.unsqueeze(0), torch.tensor(graph['ground_truth']).unsqueeze(0).long().cuda())
            avg_loss += loss
            if torch.argmax(prob) == 1:
                if graph['ground_truth'] == 1:
                    TP += 1
                elif graph['ground_truth'] == 0:
                    FP += 1
            elif torch.argmax(prob) == 0:
                if graph['ground_truth'] == 1:
                    FN += 1
                elif graph['ground_truth'] == 0:
                    TN += 1

        avg_loss /= len(val_batch)
        acc = (TP + TN) / (TP + TN + FP + FN)
        assert TP + TN + FP + FN == len(val_batch)
        precision = TP / max(TP + FP, 1)
        recall = TP / max(TP + FN, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1)

        self.log('test_loss', avg_loss.item())
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1_score)
        
    def test_step(self, val_batch, batch_idx):
        #print(val_batch[0])

        avg_loss = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for graph in val_batch:
            semantic_feature = self.relu(self.linear_before_RGCN_semantic(graph['roberta_feature']))
            try:
                entity_feature = self.dropout_layer(self.relu(self.linear_before_RGCN_entity(graph['entity_feature'])))
                node_feature = torch.cat((semantic_feature, entity_feature))
            except:
                node_feature = semantic_feature

            node_feature = self.relu(self.GatedRGCN(node_feature, graph['edge_index'], graph['edge_type']))
            node_feature = self.relu(self.GatedRGCN(node_feature, graph['edge_index'], graph['edge_type']))
            length = len(graph['roberta_feature'])
            prob = self.linear_classification(torch.mean(node_feature[:len(graph['roberta_feature'])], dim = 0))
            loss = self.CELoss(prob.unsqueeze(0), torch.tensor(graph['ground_truth']).unsqueeze(0).long().cuda())
            avg_loss += loss
            if torch.argmax(prob) == 1:
                if graph['ground_truth'] == 1:
                    TP += 1
                elif graph['ground_truth'] == 0:
                    FP += 1
            elif torch.argmax(prob) == 0:
                if graph['ground_truth'] == 1:
                    FN += 1
                elif graph['ground_truth'] == 0:
                    TN += 1

        avg_loss /= len(val_batch)
        acc = (TP + TN) / (TP + TN + FP + FN)
        assert TP + TN + FP + FN == len(val_batch)
        precision = TP / max(TP + FP, 1)
        recall = TP / max(TP + FN, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1)

        #self.log('test_loss', avg_loss.item())
        print('test_acc', acc)
        #self.log('test_precision', precision)
        #self.log('test_recall', recall)
        print('test_f1', f1_score)

test_fold = int(input('test_fold: '))

# data
dataset1 = PSPDataset(name='train', test_fold = test_fold)
test_count = 0
f = open('fold/fold_' + str(test_fold) + '.txt')
for line in f:
    test_count += 1
f.close()

dataset2 = PSPDataset(name='test', test_fold = test_fold)

train_loader = DataLoader(dataset1, batch_size=16, collate_fn=pad_collate) #important batch size
val_loader = DataLoader(dataset2, batch_size=test_count, collate_fn=pad_collate) #test all incidents

# model
model = PSPDetector(semantic_in_channels=768, entity_in_channels=200, out_channels=512, dropout=0.5, num_relations = 3)

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, max_epochs=50)
print('training begins')
trainer.fit(model, train_loader, val_loader)
trainer.test(test_dataloaders = val_loader)
