#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn import svm
import time
import pickle
import re
import threading
import pexpect
rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""


import os


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import time
import numpy as np
import gc
import sys

sys.setrecursionlimit(50000)
import pickle
import random

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
import copy
import pandas as pd

# then import my own modules
from timeit import default_timer as timer
from AttentiveFP.featurizing import graph_dict as graph
from AttentiveFP.AttentiveLayers import Fingerprint, graph_dataset, null_collate, Graph, Logger, time_to_str





SEED = 168
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
from utils import Param

sns.set()
from IPython.display import SVG, display
#import sascorer

class Attentivefp(object):
    def __init__(self, filename, **kwargs):
        self.batch_size = 50
        self.epochs = 200
        self.p_dropout = 0.2
        self.fingerprint_dim = 128
        self.weight_decay = 5  # also known as l2_regularization_lambda
        self.learning_rate = 3.5
        self.K = 2
        self.T = 2
        self.param = None
        self.data_df = None
        self.label_class = None
        self.need_gpu = True
        self.param = Param(filename,'data/tang')
        self.predict_path = 'best'
        self.weighted = 'mean'
        self.gpu = 'cpu'
        for key, value in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)

        if self.gpu == 'gpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            # cuda_aviable = torch.cuda.is_available()
            # device = torch.device(0)

    @staticmethod
    def pre_data(smiles_list):
        #print("number of all smiles: ", len(smiles_list))
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        del_smiles_list = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                atom_num_dist.append(len(mol.GetAtoms()))
                Chem.SanitizeMol(mol)
                Chem.DetectBondStereochemistry(mol, -1)
                Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
                Chem.AssignAtomChiralTagsFromStructure(mol, -1)
                remained_smiles.append(smiles)
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))                
            except:
                print('can not convert this {} smiles'.format(smiles))
                del_smiles_list.append(smiles)

        #print("number of successfully processed smiles: ", len(remained_smiles))
        return del_smiles_list

    @staticmethod
    def run_data(data_df,name):
        smiles_list = data_df.SMILES.values
        del_smiles_list = Attentivefp.pre_data(smiles_list)   #TODO: changed need debug
        data_df = data_df[~data_df.SMILES.isin(del_smiles_list)]
        smiles_list = data_df.SMILES.values
        label_list = data_df.label.values
        graph_dict = graph(smiles_list, label_list, name)
        test_df = data_df.sample(frac=0.1, random_state=SEED)
        test_smiles = test_df.SMILES.values
        training_df = data_df.drop(test_df.index)
        training_smiles = training_df.SMILES.values
        print('train smiles:{}  test smiles:{}'.format(len(training_smiles), len(test_smiles)))
        return training_smiles,test_smiles,graph_dict

    def val(self, smiles_list, graph_dict, model):
        eval_loss_list = []
        eval_loader = DataLoader(graph_dataset(smiles_list, graph_dict), self.batch_size, collate_fn=null_collate, num_workers=8,
                                 pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(eval_loader):
            atom = atom.cuda()
            bond = bond.cuda()
            bond_index = bond_index.cuda()
            mol_index = mol_index.cuda()
            label = label.cuda()
            # if self.param.normalization:
            #     label = (label - mean_list[0]) / std_list[0]

            input = model(atom, bond, bond_index, mol_index)

            # if param.multi_task:
            #     loss_ = MultiLoss()
            #     loss = loss_(input, label.view(-1, param.task_num))
            #
            # else:
            if self.param.type == 'regression':
                loss = F.l1_loss(input, label.view(-1, self.param.output_units_num), reduction='mean')
            else:
                loss = F.cross_entropy(input, label.squeeze().long(), reduction='mean')

            loss = loss.cpu().detach().numpy()
            eval_loss_list.extend([loss])
        loss = np.array(eval_loss_list).mean()
        return loss #if not self.param.normalization else np.array(eval_loss_list) * std_list[0]

    def evaluate(self):
        data_df, label_class = self.param.get_data()
        _ ,test_smiles, graph_dict = Attentivefp.run_data(data_df,self.param.name)
        fold = 5
        model_list = []
        predict_list = []
        label_list = []
        for i in range(5):
            for save_time in [
                              '2019112710', '2019112712', '2019112713', '2019112808', '2019112810', '2019112811',
                              '2019112813', '2019112814', '2019112815', '2019112816', '2019112817', '2019112818',
                              '2019112820','2019112821', '2019112900', '2019120506',
                              '2019120408',
                              ]:
                try:
                    model_list.append(
                        torch.load('saved_models/{}/fold_{}_{}_best.pt'.format(self.param.name, str(i), save_time)))
                    break
                except FileNotFoundError:
                    pass
            predict_list.append([])
            label_list.append([])
        if len(model_list) != 5:
            raise FileNotFoundError('not enough model')
        eval_loader = DataLoader(graph_dataset(test_smiles, graph_dict), self.batch_size, collate_fn=null_collate,
                                 num_workers=8,
                                 pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        for num, model in enumerate(model_list):
            model.eval()

            for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(eval_loader):
                atom = atom.cuda()
                bond = bond.cuda()
                bond_index = bond_index.cuda()
                mol_index = mol_index.cuda()
                label = label.cuda()
                mol_prediction = model(atom, bond, bond_index, mol_index)
                predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                label_list[num].extend(label.squeeze(dim=1).detach().cpu().numpy())
            #         print(predict.list)

        label = np.array(label_list).sum(axis=0) / fold

        from sklearn.linear_model import Ridge, LogisticRegression

        if self.param.type == 'regression':
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            metrics_dict_mean = {}
            for metric in self.param.metrics:
                metrics_dict_mean.update({metric: round(build_metrics_func(metric)(label, predict_mean), 4)})
            print(self.param.name + 'metrics_dict_mean :', metrics_dict_mean)
            metrics_dict_weighted = {}
            clf = Ridge(alpha=.3)
            clf.fit(np.array(predict_list).transpose(), label)
            predict_weighted = clf.predict(np.array(predict_list).transpose())
            for metric in self.param.metrics:
                metrics_dict_weighted.update({metric: round(build_metrics_func(metric)(label, predict_weighted), 4)})
            print(self.param.name + 'metrics_dict_weighted :', metrics_dict_weighted)

        elif self.param.type == 'classification':
            predict_list = softmax(predict_list,dim=2)
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            metrics_dict_mean = {}
            for metric in self.param.metrics:
                metrics_dict_mean.update({metric: round(build_metrics_func(metric)(label, softmax(predict_mean)), 4)})
            print(self.param.name + 'metrics_dict_mean :', metrics_dict_mean)

            clf = LogisticRegression()
            # clf2 = Ridge(alpha=0.3)
            clf.fit(predict_list[:, :, 1].transpose(), label)
            # clf2.fit(np.array(predict_list)[:,:,0].transpose(), -label+1)
            y_score_weighted = clf.predict_proba(predict_list[:, :, 1].transpose()).squeeze()
            metrics_dict_weighted = {}
            for metric in self.param.metrics:
                metrics_dict_weighted.update(
                    {metric: round(build_metrics_func(metric)(label, y_score_weighted), 4)})
            print(self.param.name + 'metrics_dict_weighted :', metrics_dict_weighted)
            # writer = SummaryWriter('runs/' + self.param.name + self.param.time)
            # writer.add_pr_curve('pr_curve_' + self.param.name, label, y_score_weighted[:, 1])
            # writer.add_pr_curve('pr_curve_' + self.param.name, label, y_score_mean)
            # writer.add_pr_curve('pr_curve_' + self.param.name, label, y_score_weighted[:, 1], global_step=1)
            # writer.close()

        import json
        if not os.path.exists('best/'+self.param.name):
            os.mkdir('best/'+self.param.name)
        with open('best/'+self.param.name+'/lr_weight.json','w') as f:

            f.write(json.dumps({'coef_':clf.coef_.tolist(),'intercept_':clf.intercept_.tolist()},))
        for i, model in enumerate(model_list):
            model.cpu()
            torch.save(model,'best/{}/fold_{}.pt'.format(self.param.name, str(i),))

    def predict(self,predict_smiles):
        self.param.get_data()  #TODO : 待删除 修改utils中用json保存每个任务的type，labelclass等
        del_smiles = Attentivefp.pre_data(predict_smiles)
        predict_smiles = [smiles for smiles in predict_smiles if smiles not in del_smiles]
        graph_dict = graph(predict_smiles)
        fold = 5
        model_list = []
        predict_list = []
        import json
        with open('{}/{}/lr_weight.json'.format(self.predict_path,self.param.name), 'r') as f:
            weight_dict = json.loads(f.read())
            coef_ = np.array(weight_dict['coef_'])
            intercept_ = np.array(weight_dict['intercept_'])
        for i in range(fold):
            model_list.append(torch.load('{}/{}/fold_{}.pt'.format(self.predict_path,self.param.name, str(i),)))
            predict_list.append([])
        if len(model_list) != 5:
            raise FileNotFoundError('not enough model')
        print("loader")
        eval_loader = DataLoader(graph_dataset(predict_smiles, graph_dict), self.batch_size, collate_fn=null_collate,
                                # num_workers=8,
                                 pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        for num, model in enumerate(model_list):
            if self.gpu == 'gpu':
                model.cuda()
            model.eval()

            for b, (smiles, atom, bond, bond_index, mol_index,_) in enumerate(eval_loader):
                if self.gpu == 'gpu':
                    atom = atom.cuda()
                    bond = bond.cuda()
                    bond_index = bond_index.cuda()
                    mol_index = mol_index.cuda()
                mol_prediction = model(atom, bond, bond_index, mol_index)
                if self.gpu == 'gpu':
                    predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                else:
                    predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().numpy())
        if self.param.type == 'regression':
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            predict_mean=predict_mean.tolist()
            predict_weighted = np.dot(np.array(predict_list).transpose(),coef_.transpose()) + intercept_
            if self.weighted == "mean":  #TODO:labelclass
                return predict_mean[0]
                #print(type(predict_mean[0]))
            else:
                print(predict_weighted)

        elif self.param.type == 'classification':
            predict_list = softmax(predict_list, dim=2)
            predict_mean = np.array(predict_list).sum(axis=0) / fold
            predict_mean=predict_mean.tolist()
            # print(type(predict_mean))
            # print(predict_mean)
            predict_weighted = np.dot(predict_list[:, :, 1].transpose(),coef_.transpose())+ intercept_
            predict_weighted = 1/(np.exp(-predict_weighted)+1)
            predict_weighted = np.concatenate((1-predict_weighted,predict_weighted),axis=1)
            if self.weighted == "mean":
                return predict_mean[0][1]
                #print(type(predict_mean[0]))
            else:
                print(predict_weighted)

def softmax_(x, dim=1):
    x = np.array(x, dtype=float)
    x = x.swapaxes(dim, -1)
    m = x.shape
    x = np.reshape(x, (-1, np.size(x, -1)))
    x = np.exp(x - np.reshape(np.max(x, axis=1), (-1, 1)))
    x = x / np.reshape(np.sum(x, axis=1), (-1, 1))
    x = np.reshape(x, m)
    x = x.swapaxes(dim, -1)
    return x

def softmax(x, dim=1):
    x = np.array(x, dtype=float)
    x = np.exp(x - np.expand_dims(np.max(x, axis=dim), dim))
    x = x / np.expand_dims(np.sum(x, axis=dim), dim)
    return x


def build_metrics_func(metric_name,need_max=True):
    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
        mean_squared_log_error, r2_score, median_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, \
        roc_auc_score,matthews_corrcoef
    func = locals()[metric_name]
    if metric_name in ['accuracy_score','f1_score','recall_score','precision_score','matthews_corrcoef']:
        if need_max:
            return lambda x,y:func(x,np.argmax(y,axis=1))
        else:
            return lambda x,y:func(x,np.round(y))


    if metric_name in ['average_precision_score','roc_auc_score',]:
        if need_max:
            return lambda x,y:func(x,y[:,1])
        else:
            return lambda x,y:func(x,y)

    return locals()[metric_name]

# Fingerprint
# for _, _, file_list in os.walk('data/tang'):
#     for file in file_list:
#         filename = file.split('.')[0]
#         # if filename not in ['M_CYPPro_I']:
#         #     continue
#         if filename in ['test', 'bioinformatics2019']:
#             continue
#         model1 = Attentivefp(filename)
#         model1.predict()
#         # model1.evaluate()
#         print(filename)






class no_sulphur():
    """Scores structures based on not containing sulphur."""

    kwargs = []

    def __init__(self):
        pass
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            has_sulphur = [16 not in [atom.GetAtomicNum() for atom in mol.GetAtoms()]]
            return float(has_sulphur)
        return 0.0

class no_bias():
    """Scores structures based on not containing sulphur."""

    def __init__(self):
        pass
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            has_sulphur = [16 not in [atom.GetAtomicNum() for atom in mol.GetAtoms()]]
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0



class tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    kwargs = ["k", "query_structure"]
    k = 0.7
    query_structure = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F"

    def __init__(self):
        query_mol = Chem.MolFromSmiles(self.query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
            score = DataStructs.TanimotoSimilarity(self.query_fp, fp)
            score = min(score, self.k) / self.k
            return float(score)
        return 0.0

class activity_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/clf.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = activity_model.fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return 0.0

    @classmethod
    def fingerprints_from_mol(cls, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx,v in fp.GetNonzeroElements().items():
            nidx = idx%size
            nfp[0, nidx] += int(v)
        return nfp

class activity_model_tanimoto():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path", "k", "query_structure"]
    clf_path = 'data/clf.pkl'
    k = 0.7
    query_structure = "CCCN(CCN1CCN(c2ccccc2OC)CC1)C1CCc2c(O)cccc2C1"

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)
        query_mol = Chem.MolFromSmiles(self.query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp1 = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
            score1 = DataStructs.TanimotoSimilarity(self.query_fp, fp1)
            score1 = 0.5 * min(score1, self.k) / self.k
            fp2 = activity_model.fingerprints_from_mol(mol)
            score2 = 0.5* self.clf.predict_proba(fp2)[:, 1]
            score = score1 + score2
            return float(score), 2*float(score1), 2*float(score2)
        return 0.0, 0.0, 0.0



    @classmethod
    def fingerprints_from_mol(cls, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx,v in fp.GetNonzeroElements().items():
            nidx = idx%size
            nfp[0, nidx] += int(v)
        return nfp



class cs286():
    """Scores based on an ECFP classifier for activity."""

    # kwargs = ["clf_path", "k", "query_structure"]
    # clf_path = 'data/clf.pkl'
    # k = 0.7
    #query_structure = "CCCN(CCN1CCN(c2ccccc2OC)CC1)C1CCc2c(O)cccc2C1"

    def __init__(self):
        #query_mol = Chem.MolFromSmiles(self.query_structure)
        self.param = Param(filename,'data/tang')
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        
        if mol:
            self.param.get_data()  #TODO : 待删除 修改utils中用json保存每个任务的type，labelclass等
            del_smiles = Attentivefp.pre_data(smile)
            predict_smiles = [smiles for smiles in smile if smiles not in del_smiles]
            graph_dict = graph(smile)
            fold = 5
            model_list = []
            predict_list = []
            import json
            with open('{}/{}/lr_weight.json'.format(self.predict_path,self.param.name), 'r') as f:
                weight_dict = json.loads(f.read())
                coef_ = np.array(weight_dict['coef_'])
                intercept_ = np.array(weight_dict['intercept_'])
            for i in range(fold):
                model_list.append(torch.load('{}/{}/fold_{}.pt'.format(self.predict_path,self.param.name, str(i),)))
                predict_list.append([])
            if len(model_list) != 5:
                raise FileNotFoundError('not enough model')
            print("loader")
            eval_loader = DataLoader(graph_dataset(smile, graph_dict), self.batch_size, collate_fn=null_collate,
                                    # num_workers=8,
                                    pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
            for num, model in enumerate(model_list):
                if self.gpu == 'gpu':
                    model.cuda()
                model.eval()

                for b, (smiles, atom, bond, bond_index, mol_index,_) in enumerate(eval_loader):
                    if self.gpu == 'gpu':
                        atom = atom.cuda()
                        bond = bond.cuda()
                        bond_index = bond_index.cuda()
                        mol_index = mol_index.cuda()
                    mol_prediction = model(atom, bond, bond_index, mol_index)
                    if self.gpu == 'gpu':
                        predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                    else:
                        predict_list[num].extend(mol_prediction.squeeze(dim=1).detach().numpy())


            predict_mean1 = np.array(predict_list).sum(axis=0) / fold
            predict_mean1=predict_mean1.tolist()
            score2= predict_mean1[0]
        
            predict_list2 = softmax(predict_list, dim=2)
            predict_mean2 = np.array(predict_list2).sum(axis=0) / fold
            predict_mean2=predict_mean2.tolist()
            score1= predict_mean2[0][1]
            score = 0.5*score1 + 0.5*score2
            return float(score), float(score1), float(score2)
        return 0.0, 0.0, 0.0
        
        # import argparse
        # parser1 = argparse.ArgumentParser()
        #     # parser.add_argument('-f', '--function', type=str, choices=['train','evaluate','predict'], required=False,default='predict')
        #     # parser.add_argument('-n', '--name', type=str, required=True, help="task name")
        #     # parser.add_argument('-g', '--gpu', type=str, choices=['gpu','cpu'],required=False, default='cpu')
        #     # parser.add_argument('-w', '--weighted', type=str, choices=['mean','weighted'],required=False, default='mean')
        #     # parser.add_argument('-s', '--smiles', type=str, required=False, )
        # args1 = parser1.parse_args()
        # args1.weighted='mean'
        # args1.gpu="gpu"
        # arg1.name1="A_BBB_I"
        # arg1.name2="A_Caco2_I"
        # arg1.function="predict"
        # afp_model1 = Attentivefp("CB1",weighted='mean',gpu="gpu")
        # afp_model2 = Attentivefp("solubility",weighted='mean',gpu="gpu")

            # score1=getattr(afp_model1, "predict")([smile])
            # score2=getattr(afp_model2, "predict")([smile])




class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""

        self.proc = pexpect.spawn('./multiprocess.py ' + scoring_function,
                                  encoding='utf-8')

        print(self.is_alive())

    def __call__(self, smile, index, result_list):
        self.proc.sendline(smile)
        output = self.proc.expect([re.escape(smile) + " 1\.0+|[0]\.[0-9]+", 'None', pexpect.TIMEOUT])
        if output is 0:
            score = float(self.proc.after.lstrip(smile + " "))
        elif output in [1, 2]:
            score = 0.0
        result_list[index] = score

    def is_alive(self):
        return self.proc.isalive()

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None):
        self.n = num_processes
        self.workers = [Worker(scoring_function=scoring_function) for _ in range(num_processes)]

    def alive_workers(self):
        return [i for i, worker in enumerate(self.workers) if worker.is_alive()]

    def __call__(self, smiles):
        scores = [0 for _ in range(len(smiles))]
        smiles_copy = [smile for smile in smiles]
        while smiles_copy:
            alive_procs = self.alive_workers()
            if not alive_procs:
               raise RuntimeError("All subprocesses are dead, exiting.")
            # As long as we still have SMILES to score
            used_threads = []
            # Threads name corresponds to the index of the worker, so here
            # we are actually checking which workers are busy
            for t in threading.enumerate():
                # Workers have numbers as names, while the main thread cant
                # be converted to an integer
                try:
                    n = int(t.name)
                    used_threads.append(n)
                except ValueError:
                    continue
            free_threads = [i for i in alive_procs if i not in used_threads]
            for n in free_threads:
                if smiles_copy:
                    # Send SMILES and what index in the result list the score should be inserted at
                    smile = smiles_copy.pop()
                    idx = len(smiles_copy)
                    t = threading.Thread(target=self.workers[n], name=str(n), args=(smile, idx, scores))
                    t.start()
            time.sleep(0.01)
        for t in threading.enumerate():
            try:
                n = int(t.name)
                t.join()
            except ValueError:
                continue
        return np.array(scores, dtype=np.float32)

class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()
        self.batch_size = 64
        self.epochs = 200
        self.p_dropout = 0.2
        self.fingerprint_dim = 128
        self.weight_decay = 5  # also known as l2_regularization_lambda
        self.learning_rate = 3.5
        self.K = 2
        self.T = 2
        self.param = None
        self.data_df = None
        self.label_class = None
        self.need_gpu = True
        filename1="CB1"
        filename2="solubility"
        filename3="A_BBB_I"
        filename4="T_hERG_II"
        self.param1 = Param(filename1,'data/tang')
        self.param2 = Param(filename2,'data/tang')
        self.param3 = Param(filename3,'data/tang')
        self.param4 = Param(filename4,'data/tang')
        self.predict_path = 'best'
        self.weighted = 'mean'
        self.gpu = 'cpu'
        # for key, value in kwargs.items():
        #     if hasattr(self,key):
        #         setattr(self,key,value)

        if self.gpu == 'gpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            # cuda_aviable = torch.cuda.is_available()
            # device = torch.device(0)
    def __call__(self, smiles_0):
        def softmax_(x, dim=1):
            x = np.array(x, dtype=float)
            x = x.swapaxes(dim, -1)
            m = x.shape
            x = np.reshape(x, (-1, np.size(x, -1)))
            x = np.exp(x - np.reshape(np.max(x, axis=1), (-1, 1)))
            x = x / np.reshape(np.sum(x, axis=1), (-1, 1))
            x = np.reshape(x, m)
            x = x.swapaxes(dim, -1)
            return x

        def softmax(x, dim=1):
            x = np.array(x, dtype=float)
            x = np.exp(x - np.expand_dims(np.max(x, axis=dim), dim))
            x = x / np.expand_dims(np.sum(x, axis=dim), dim)
            return x


        def build_metrics_func(metric_name,need_max=True):
            from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
                mean_squared_log_error, r2_score, median_absolute_error, r2_score
            from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, \
                roc_auc_score,matthews_corrcoef
            func = locals()[metric_name]
            if metric_name in ['accuracy_score','f1_score','recall_score','precision_score','matthews_corrcoef']:
                if need_max:
                    return lambda x,y:func(x,np.argmax(y,axis=1))
                else:
                    return lambda x,y:func(x,np.round(y))


            if metric_name in ['average_precision_score','roc_auc_score',]:
                if need_max:
                    return lambda x,y:func(x,y[:,1])
                else:
                    return lambda x,y:func(x,y)

            return locals()[metric_name]
        smiles=[]
        empty_index=[]
        for i in range(len(smiles_0)):
            mol = Chem.MolFromSmiles(smiles_0[i])
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    Chem.DetectBondStereochemistry(mol, -1)
                    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
                    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
                    smiles.append(smiles_0[i])
                except:
                    print('can not convert this {} smiles'.format(smiles_0[i]))
                    empty_index.append(i)
            else:
                empty_index.append(i)
            
            # mol = Chem.MolFromSmiles(smiles_0[i])
            # if mol:
            #     smiles.append(smiles_0[i])
            # else:
            #     empty_index.append(i)
        #print(smiles)
        #print(empty_index)
        self.param1.get_data()  #TODO : 待删除 修改utils中用json保存每个任务的type，labelclass等
        self.param2.get_data()
        del_smiles = Attentivefp.pre_data(smiles)
        predict_smiles = [smile for smile in smiles if smile not in del_smiles]
        #print(predict_smiles)
        graph_dict = graph(smiles)
        #print(graph_dict)
        fold = 5
        model_list1 = []
        predict_list1 = []
        model_list2 = []
        predict_list2 = []
        import json
        with open('{}/{}/lr_weight.json'.format(self.predict_path,self.param1.name), 'r') as f1:
            weight_dict1 = json.loads(f1.read())
            coef_1 = np.array(weight_dict1['coef_'])
            intercept_1 = np.array(weight_dict1['intercept_'])
        with open('{}/{}/lr_weight.json'.format(self.predict_path,self.param2.name), 'r') as f2:
            weight_dict2 = json.loads(f2.read())
            coef_2 = np.array(weight_dict2['coef_'])
            intercept_2 = np.array(weight_dict2['intercept_'])
        for i in range(fold):
            model_list1.append(torch.load('{}/{}/fold_{}.pt'.format(self.predict_path,self.param1.name, str(i),)))
            predict_list1.append([])
            model_list2.append(torch.load('{}/{}/fold_{}.pt'.format(self.predict_path,self.param2.name, str(i),)))
            predict_list2.append([])
        if len(model_list1) != 5 or len(model_list2) != 5:
            raise FileNotFoundError('not enough model')
        #print("loader")
        eval_loader = DataLoader(graph_dataset(smiles, graph_dict), self.batch_size, collate_fn=null_collate,
                                # num_workers=8,
                                pin_memory=True, shuffle=False, worker_init_fn=np.random.seed(SEED))
        
        for num, model in enumerate(model_list1):
            if self.gpu == 'gpu':
                model.cuda()
            model.eval()
            
            for b, (smiles, atom, bond, bond_index, mol_index,_) in enumerate(eval_loader):
                #print(smiles)
                if self.gpu == 'gpu':
                    atom = atom.cuda()
                    bond = bond.cuda()
                    bond_index = bond_index.cuda()
                    mol_index = mol_index.cuda()
                mol_prediction = model(atom, bond, bond_index, mol_index)
                if self.gpu == 'gpu':
                    predict_list1[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                else:
                    predict_list1[num].extend(mol_prediction.squeeze(dim=1).detach().numpy())
        for num, model in enumerate(model_list2):
            if self.gpu == 'gpu':
                model.cuda()
            model.eval()

            for b, (smiles, atom, bond, bond_index, mol_index,_) in enumerate(eval_loader):
                if self.gpu == 'gpu':
                    atom = atom.cuda()
                    bond = bond.cuda()
                    bond_index = bond_index.cuda()
                    mol_index = mol_index.cuda()
                mol_prediction = model(atom, bond, bond_index, mol_index)
                if self.gpu == 'gpu':
                    predict_list2[num].extend(mol_prediction.squeeze(dim=1).detach().cpu().numpy())
                else:
                    predict_list2[num].extend(mol_prediction.squeeze(dim=1).detach().numpy())

        score,score1,score2=[],[],[]
        #print(predict_list1)
        predict_list1 = softmax(predict_list1, dim=2)
        #print(predict_list1)
        predict_mean1 = np.array(predict_list1).sum(axis=0) / fold     ###classification
        #print(predict_mean1)
        score1=[float(score[1]) for score in predict_mean1]
        predict_mean2 = np.array(predict_list2).sum(axis=0) / fold   ###regression
        #print(predict_mean2)
        score2=[float(score) for score in predict_mean2]
        #print(score1)
        #print(score2)
        score=[0.5*score1[i]+0.5*score2[i]  for i in range(len(score1))]
        for index in empty_index:
            #print(index)
            score.insert(index,'0.0')
        #print(score)
        

        
        #scores = [self.scoring_function(smile)[0] for smile in smiles]
        # scores1 = [self.scoring_function(smile)[1] for smile in smiles]
        # scores2 = [self.scoring_function(smile)[2] for smile in smiles]
        return np.array(score, dtype=np.float32), np.array(score1, dtype=np.float32), np.array(score2, dtype=np.float32)
        #return score, score1, score2

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [no_sulphur, tanimoto, activity_model, activity_model_tanimoto, cs286, no_bias]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    for k, v, j in kwargs.items():
        if k in scoring_function_class.kwargs:
            setattr(scoring_function_class, k, v, j)

    if num_processes == 0:
        return Singleprocessing(scoring_function=scoring_function_class)
    return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes)
