import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
import h5py
from tqdm import tqdm
import json
sys.setrecursionlimit(500000000)
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train"):
        self.train_path = "train_process.txt"
        self.val_path = "dev_process.txt"  # "validD.txt"
        self.test_path = "test_process.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50
        self.ruledict = pickle.load(open("rule.pkl", "rb"))
        self.ruledict["start -> Module"] = len(self.ruledict)
        self.ruledict["start -> copyword"] = len(self.ruledict)
        self.rrdict = {}
        for x in self.ruledict:
            self.rrdict[self.ruledict[x]] = x
        if not os.path.exists("nl_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        #print(self.Nl_Voc)
        if dataName == "train":
            if os.path.exists("data.pkl"):
                self.data = pickle.load(open("data.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.train_path, "r", encoding='utf-8'))
        elif dataName == "val":
            if os.path.exists("valdata.pkl"):
                self.data = pickle.load(open("valdata.pkl", "rb"))
                self.nl = pickle.load(open("valnl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.val_path, "r", encoding='utf-8'))
        else:
            if os.path.exists("testdata.pkl"):
                self.data = pickle.load(open("testdata.pkl", "rb"))
                #self.code = pickle.load(open("testcode.pkl", "rb"))
                self.nl = pickle.load(open("testnl.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.test_path, "r", encoding='utf-8'))

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
        self.Nl_Voc["<emptynode>"] = len(self.Nl_Voc)
        self.Code_Voc["<emptynode>"] = len(self.Code_Voc)

    def init_dic(self):
        print("initVoc")
        f = open(self.train_path, "r", encoding='utf-8')
        lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        nls = []
        rules = []
        for i in tqdm(range(int(len(lines) / 5))):
            data = lines[5 * i].strip().lower().split()
            nls.append(data)
            rulelist = lines[5 * i + 1].strip().split()
            tmp = []
            for x in rulelist:
                if int(x) >= 10000:
                    tmp.append(data[int(x) - 10000])
            rules.append(tmp)
        f.close()
        nl_voc = VocabEntry.from_corpus(nls, size=50000, freq_cutoff=0)
        code_voc = VocabEntry.from_corpus(rules, size=50000, freq_cutoff=10)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id
        for x in self.ruledict:
            lst = x.strip().lower().split()
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.Code_Voc:
                    self.Code_Voc[y] = len(self.Code_Voc)
            #rules.append([lst[0]] + lst[2:])
        #print(self.Code_Voc)
        assert("module" in self.Code_Voc)
        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
        print(maxNlLen, maxCodeLen, maxCharLen)
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def preProcessData(self, dataFile):
        lines = dataFile.readlines()
        inputNl = []
        inputNlChar = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputDepth = []
        nls = []
        for i in tqdm(range(int(len(lines) / 5))):
            child = {}
            nl = lines[5 * i].lower().strip().split()
            nls.append(nl)
            inputparent = lines[5 * i + 2].strip().split()
            inputres = lines[5 * i + 1].strip().split()
            depth = lines[5 * i + 3].strip().split()
            parentname = lines[5 * i + 4].strip().lower().split()
            inputad = np.zeros([self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])
            for i in range(min(self.Nl_Len, len(nl))):
                for j in range(min(self.Nl_Len, len(nl))):
                    inputad[i, j] = 1
            inputrule = [self.ruledict["start -> Module"]]
            for j in range(len(inputres)):
                inputres[j] = int(inputres[j])
                #depth[j] = int(depth[j])
                inputparent[j] = int(inputparent[j]) + 1
                child.setdefault(inputparent[j], []).append(j + 1)
                if inputres[j] >= 10000:
                    inputres[j] = len(self.ruledict) + inputres[j] - 10000
                    if j + 1 < self.Code_Len:
                        inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> copyword'])
                else:
                    inputrule.append(inputres[j])
                if inputres[j] - len(self.ruledict) >= self.Nl_Len:
                    print(inputres[j] - len(self.ruledict))
                if j + 1 < self.Code_Len:
                    inputad[self.Nl_Len + j + 1, self.Nl_Len + inputparent[j]] = 1
            depth = [self.pad_seq([1], 40)]
            for j in range(len(inputres)):
                tmp = []
                ids = child[inputparent[j]].index(j + 1) + 1
                tmp.append(ids)
                tmp.extend(depth[inputparent[j]])
                tmp = self.pad_seq(tmp, 40)
                depth.append(tmp)
            depth = self.pad_list(depth, self.Code_Len, 40)
            #inputrule = [self.ruledict["start -> Module"]] + inputres
            #depth = self.pad_seq([1] + depth, self.Code_Len)
            inputnls = self.Get_Em(nl, self.Nl_Voc)
            inputNl.append(self.pad_seq(inputnls, self.Nl_Len))
            inputnlchar = self.Get_Char_Em(nl)
            for j in range(len(inputnlchar)):
                inputnlchar[j] = self.pad_seq(inputnlchar[j], self.Char_Len)
            inputnlchar = self.pad_list(inputnlchar, self.Nl_Len, self.Char_Len)
            inputNlChar.append(inputnlchar)
            inputruleparent = self.pad_seq(self.Get_Em(["start"] + parentname, self.Code_Voc), self.Code_Len)
            inputrulechild = []
            for x in inputrule:
                if x >= len(self.rrdict):
                    inputrulechild.append(self.pad_seq(self.Get_Em(["copyword"], self.Code_Voc), self.Char_Len))
                else:
                    rule = self.rrdict[x].strip().lower().split()
                    inputrulechild.append(self.pad_seq(self.Get_Em(rule[2:], self.Code_Voc), self.Char_Len))

            inputparentpath = []
            for j in range(len(inputres)):
                if inputres[j] in self.rrdict:
                    tmppath = [self.rrdict[inputres[j]].strip().lower().split()[0]]
                    assert(tmppath[0] == parentname[j].lower())
                else:
                    tmppath = [parentname[j].lower()]
                '''siblings = child[inputparent[j]]
                for x in siblings:
                    if x == j + 1:
                        break
                    tmppath.append(parentname[x - 1])'''
                curr = inputparent[j]
                while curr != 0:
                    rule = self.rrdict[inputres[curr - 1]].strip().lower().split()[0]
                    tmppath.append(rule)
                    curr = inputparent[curr - 1]
                inputparentpath.append(self.pad_seq(self.Get_Em(tmppath, self.Code_Voc), 10))
            inputrule = self.pad_seq(inputrule, self.Code_Len)
            inputres = self.pad_seq(inputres, self.Code_Len)
            tmp = [self.pad_seq(self.Get_Em(['start'], self.Code_Voc), 10)] + inputparentpath
            inputrulechild = self.pad_list(tmp, self.Code_Len, 10)
            inputRuleParent.append(inputruleparent)
            inputRuleChild.append(inputrulechild)
            inputRes.append(inputres)
            inputRule.append(inputrule)
            inputparent = [0] + inputparent
            inputParent.append(inputad)
            inputParentPath.append(self.pad_list(inputparentpath, self.Code_Len, 10))
            inputDepth.append(depth)
        batchs = [inputNl, inputNlChar, inputRule, inputRuleParent, inputRuleChild, inputRes, inputParent, inputParentPath, inputDepth]
        self.data = batchs
        self.nls = nls
        #self.code = codes
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("nl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
            open("valnl.pkl", "wb").write(pickle.dumps(nls))
        if self.dataName == "test":
            open("testdata.pkl", "wb").write(pickle.dumps(batchs))
            #open("testcode.pkl", "wb").write(pickle.dumps(self.code))
            open("testnl.pkl", "wb").write(pickle.dumps(self.nls))
        return batchs

    def __getitem__(self, offset):
        ans = []
        '''if self.dataName == "train":
            h5f = h5py.File("data.h5", 'r')
        if self.dataName == "val":
            h5f = h5py.File("valdata.h5", 'r')
        if self.dataName == "test":
            h5f = h5py.File("testdata.h5", 'r')'''
        for i in range(len(self.data)):
            d = self.data[i][offset]
            '''if i == 6:
                #print(self.data[i][offset])
                tmp = np.eye(self.Code_Len)[d]
                #print(tmp.shape)
                tmp = np.concatenate([tmp, np.zeros([self.Code_Len, self.Code_Len])], axis=0)[:self.Code_Len,:]#self.pad_list(tmp, self.Code_Len, self.Code_Len)
                ans.append(np.array(tmp))
            else:'''
            ans.append(np.array(d))
        return ans
    def __len__(self):
        return len(self.data[0])
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.sibiling = None
    
#dset = SumDataset(args)
