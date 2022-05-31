import torch
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
import wandb
from copy import deepcopy
import pickle
from ScheduledOptim import *
import sys
from Radam import RAdam
import torch.nn.functional as F
#from pythonBottom.run import finetune
#from pythonBottom.run import pre
#wandb.init("sql")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':50,
    'CodeLen':200,
    'batch_size':32,
    'embedding_size':312,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'num_steps':50,
    'rulenum':10,
    'seed':0
})
#os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
def save_model(model, dirs='checkpointSearch/'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    #cprint(dirs)
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt'))
use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans
def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    return ans
def getRulePkl(vds):
    inputruleparent = []
    inputrulechild = []
    for i in range(len(vds.ruledict)):
        rule = vds.rrdict[i].strip().lower().split()
        inputrulechild.append(vds.pad_seq(vds.Get_Em(rule[2:], vds.Code_Voc), vds.Char_Len))
        inputruleparent.append(vds.Code_Voc[rule[0].lower()])
    return np.array(inputruleparent), np.array(inputrulechild)
def evalacc(model, dev_set):
    antimask = gVar(getAntiMask(args.CodeLen))
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=22,
                                              shuffle=False, drop_last=True, num_workers=1)
    model = model.eval()
    accs = []
    tcard = []
    antimask2 = antimask.unsqueeze(0).repeat(22, 1, 1).unsqueeze(1)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(2, 1).long()
    for devBatch in tqdm(devloader):
        for i in range(len(devBatch)):
            devBatch[i] = gVar(devBatch[i])
        with torch.no_grad():
            _, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[6], devBatch[7], devBatch[8], tmpf, tmpc, tmpindex, rulead, antimask2, devBatch[5])
            pred = pre.argmax(dim=-1)
            resmask = torch.gt(devBatch[5], 0)
            acc = (torch.eq(pred, devBatch[5]) * resmask).float()#.mean(dim=-1)
            predres = (1 - acc) * pred.float() * resmask.float()
            accsum = torch.sum(acc, dim=-1)
            resTruelen = torch.sum(resmask, dim=-1).float()
            print(torch.eq(accsum, resTruelen))
            cnum = (torch.eq(accsum, resTruelen)).sum().float()
            acc = acc.sum(dim=-1) / resTruelen
            accs.append(acc.mean().item())
            tcard.append(cnum.item())
                        #print(devBatch[5])
                        #print(predres)
    tnum = np.sum(tcard)
    acc = np.mean(accs)
                #wandb.log({"accuracy":acc})
    return acc, tnum
def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set = SumDataset(args, "train")
    print(len(train_set.rrdict))
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    tmpindex = gVar(np.arange(len(train_set.ruledict))).unsqueeze(0).repeat(2, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict) + args.NlLen
    dev_set = SumDataset(args, "val")
    test_set = SumDataset(args, "test")
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
    model = Decoder(args)
    #load_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, n_warmup_steps=4000)
    maxAcc = 0
    maxC = 0
    maxAcc2 = 0
    maxC2 = 0
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    antimask = gVar(getAntiMask(args.CodeLen))
    #model.to()
    for epoch in range(100000):
        j = 0
        for dBatch in tqdm(data_loader):
            if j % 3000 == 0:
                acc, tnum = evalacc(model, dev_set)
                acc2, tnum2 = evalacc(model, test_set)
                print("for dev " + str(acc) + " " + str(tnum) + " max is " + str(maxC))
                print("for test " + str(acc2) + " " + str(tnum2) + " max is " + str(maxC2))
                if maxC < tnum or maxC == tnum and maxAcc < acc:
                    maxC = tnum
                    maxAcc = acc
                    print("find better acc " + str(maxAcc))
                    save_model(model.module, 'checkModel%s/'%args.seed)
                if maxC2 < tnum2 or maxC2 == tnum2 and maxAcc2 < acc2:
                    maxC2 = tnum2
                    maxAcc2 = acc2
                    print("find better acc " + str(maxAcc2))
                    save_model(model.module, "test%s/"%args.seed)
            #exit(0)
            antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3], dBatch[4], dBatch[6], dBatch[7], dBatch[8], tmpf, tmpc, tmpindex, rulead, antimask2, dBatch[5])
            loss = torch.mean(loss) + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            j += 1
class Node:
    def __init__(self, name, d):
        self.name = name
        self.depth = d
        self.father = None
        self.child = []
        self.sibiling = None
        self.expanded = False
        self.fatherlistID = 0
class SearchNode:
    def __init__(self, ds):
        self.state = [ds.ruledict["start -> Module"]]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.root = Node("Module", 2)
        self.inputparent = ["start"]
        self.parent = np.zeros([args.NlLen + args.CodeLen, args.NlLen + args.CodeLen])
        #self.parent[args.NlLen]
        self.expanded = None
        self.ruledict = ds.rrdict
        self.expandedname = []
        self.depth = [1]
        for x in ds.ruledict:
            self.expandedname.append(x.strip().split()[0])
        self.everTreepath = []
    def selcetNode(self, root):
        if not root.expanded and root.name in self.expandedname and root.name != "list" and self.state[root.fatherlistID] < len(self.ruledict):
            return root
        else:
            for x in root.child:
                ans = self.selcetNode(x)
                if ans:
                    return ans
            if root.name == "list" and root.expanded == False:
                return root
        return None
    def selectExpandedNode(self):
        self.expanded = self.selcetNode(self.root)
    def getRuleEmbedding(self, ds, nl):
        inputruleparent = []
        inputrulechild = []
        for x in self.state:
            if x >= len(ds.rrdict):
                inputruleparent.append(ds.Get_Em(["value"], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(["copyword"], ds.Code_Voc), ds.Char_Len))
            else:
                rule = ds.rrdict[x].strip().lower().split()
                inputruleparent.append(ds.Get_Em([rule[0]], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(rule[2:], ds.Code_Voc), ds.Char_Len))
        inputrule = ds.pad_seq(self.state, ds.Code_Len)
        inputrulechild = ds.pad_list(inputrulechild, ds.Code_Len, ds.Char_Len)
        inputruleparent = ds.pad_seq(ds.Get_Em(self.inputparent, ds.Code_Voc), ds.Code_Len)
        inputdepth = ds.pad_seq(self.depth, ds.Code_Len)
        return inputrule, inputrulechild, inputruleparent, inputdepth
    def getTreePath(self, ds):
        tmppath = [self.expanded.name.lower()]
        node = self.expanded.father
        while node:
            tmppath.append(node.name.lower())
            node = node.father
        tmp = ds.pad_seq(ds.Get_Em(tmppath, ds.Code_Voc), 10)
        self.everTreepath.append(tmp)
        return ds.pad_list(self.everTreepath, ds.Code_Len, 10)
    def applyrule(self, rule, nl):
        if rule >= len(self.ruledict):
            if rule - len(self.ruledict) >= len(nl):
                return False
            if self.expanded.depth + 1 >= 40:
                nnode = Node(nl[rule - len(self.ruledict)], 39)
            else:
                nnode = Node(nl[rule - len(self.ruledict)], self.expanded.depth + 1)
            self.expanded.child.append(nnode)
            nnode.father = self.expanded
            nnode.fatherlistID = len(self.state)
        else:
            rules = self.ruledict[rule]
            if rules.strip().split()[0] != self.expanded.name:
                #print(self.expanded.name)
                return False
            #assert(rules.strip().split()[0] == self.expanded.name)
            if rules == self.expanded.name + " -> End ":
                self.expanded.expanded = True
            else:
                for x in rules.strip().split()[2:]:
                    if self.expanded.depth + 1 >= 40:
                        nnode = Node(x, 39)
                    else:
                        nnode = Node(x, self.expanded.depth + 1)                   
                    #nnode = Node(x, self.expanded.depth + 1)
                    self.expanded.child.append(nnode)
                    nnode.father = self.expanded
                    nnode.fatherlistID = len(self.state)
        #self.parent.append(self.expanded.fatherlistID)
        self.parent[args.NlLen + len(self.depth), args.NlLen + self.expanded.fatherlistID] = 1
        if rule >= len(self.ruledict):
            self.parent[args.NlLen + len(self.depth), rule - len(self.ruledict)] = 1
        self.state.append(rule)
        self.inputparent.append(self.expanded.name.lower())
        self.depth.append(self.expanded.depth)
        if self.expanded.name != "list":
            self.expanded.expanded = True
        return True
    def printTree(self, r):
        s = r.name + " "#print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s
    def getTreestr(self):
        return self.printTree(self.root)

        
beamss = []
def BeamSearch(inputnl, vds, model, beamsize, batch_size, k):
    args.batch_size = len(inputnl[0])
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNode(vds)]
        index = 0
        antimask = gVar(getAntiMask(args.CodeLen))
        endnum = {}
        continueSet = {}
        while True:
            print(index)
            tmpbeam = {}
            ansV = {}
            if len(endnum) == args.batch_size:
                break
            if index >= args.CodeLen:
                break
            for p in range(beamsize):
                tmprule = []
                tmprulechild = []
                tmpruleparent = []
                tmptreepath = []
                tmpAd = []
                validnum = []
                tmpdepth = []
                for i in range(args.batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    #print(x.getTreestr())
                    x.selectExpandedNode()
                    if x.expanded == None or len(x.state) >= args.CodeLen:
                        ansV.setdefault(i, []).append(x)
                    else:
                        #print(x.expanded.name)
                        validnum.append(i)
                        a, b, c, d = x.getRuleEmbedding(vds, vds.nl[args.batch_size * k + i])
                        tmprule.append(a)
                        tmprulechild.append(b)
                        tmpruleparent.append(c)
                        tmptreepath.append(x.getTreePath(vds))
                        #tmp = np.eye(vds.Code_Len)[x.parent]
                        #tmp = np.concatenate([tmp, np.zeros([vds.Code_Len, vds.Code_Len])], axis=0)[:vds.Code_Len,:]#self.pad_list(tmp, self.Code_Len, self.Code_Len)
                        tmpAd.append(x.parent)
                        tmpdepth.append(d)
                #print("--------------------------")
                if len(tmprule) == 0:
                    continue
                batch_size = len(tmprule)
                antimasks = antimask.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1)
                tmprule = np.array(tmprule)
                tmprulechild = np.array(tmprulechild)
                tmpruleparent = np.array(tmpruleparent)
                tmptreepath = np.array(tmptreepath)
                tmpAd = np.array(tmpAd)
                tmpdepth = np.array(tmpdepth)
                '''print(inputnl[3][:index + 1], tmprule[:index + 1])
                assert(np.array_equal(inputnl[3][0][:index + 1], tmprule[0][:index + 1]))
                assert(np.array_equal(inputnl[4][0][:index + 1], tmpruleparent[0][:index + 1]))
                assert(np.array_equal(inputnl[5][0][:index + 1], tmprulechild[0][:index + 1]))
                assert(np.array_equal(inputnl[6][0][:index + 1], tmpAd[0][:index + 1]))
                assert(np.array_equal(inputnl[7][0][:index + 1], tmptreepath[0][:index + 1]))
                assert(np.array_equal(inputnl[8][0][:index + 1], tmpdepth[0][:index + 1]))'''
                result = model(gVar(inputnl[0][validnum]), gVar(inputnl[1][validnum]), gVar(tmprule), gVar(tmpruleparent), gVar(tmprulechild), gVar(tmpAd), gVar(tmptreepath), gVar(tmpdepth), antimasks, None, "test")
                results = result.data.cpu().numpy()
                #print(result, inputCode)
                currIndex = 0
                for j in range(args.batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = beamsize
                    result = np.negative(results[currIndex, index])
                    currIndex += 1
                    cresult = np.negative(result)
                    indexs = np.argsort(result)
                    for i in range(tmpbeamsize):
                        if tmpbeamsize >= 20:
                            break
                        copynode = deepcopy(x)
                        #if indexs[i] >= len(vds.rrdict):
                            #print(cresult[indexs[i]])
                        c = copynode.applyrule(indexs[i], vds.nl[args.batch_size * k + j])
                        if not c:
                            tmpbeamsize += 1
                            continue
                        copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append(copynode)
                    #print(tmpbeam[0].prob)
            for i in range(args.batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(args.batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append(x)
                    beams[j] = sorted(tmpbeam[j], key=lambda x: x.prob, reverse=True)[:beamsize]
            index += 1
        for p in range(beamsize):
            beam = []
            nls = []
            for i in range(len(beams)):
                if p >= len(beams[i]):
                    beam.append(beams[i][len(beams[i]) - 1])
                else:
                    beam.append(beams[i][p])
                nls.append(vds.nl[args.batch_size * k + i])
            finetune(beam, k, nls, args.batch_size)
        for i in range(len(beams)):
            beamss.append(deepcopy(beams[i]))
            

        for i in range(len(beams)):
            mans = -1000000
            lst = beams[i]
            tmpans = 0
            for y in lst:
                #print(y.getTreestr())
                if y.prob > mans:
                    mans = y.prob
                    tmpans = y
            beams[i] = tmpans
        open("beams.pkl", "wb").write(pickle.dumps(beamss))
        return beams
        #return beams
def test():
    pre()
    dev_set = SumDataset(args, "test")
    print(len(dev_set))
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict) + args.NlLen
    args.batch_size = 66
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)
    if torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0)
    model = model.eval()
    load_model(model)
    f = open("outval.txt", "w")
    index = 0 
    for x in tqdm(devloader):
        '''if index == 0:
            index += 1
            continue'''
        ans = BeamSearch((x[0], x[1], x[5], x[2], x[3], x[4], x[6], x[7], x[8]), dev_set, model, 15, args.batch_size, index)
        index += 1
        for i in range(args.batch_size):
            beam = ans[i]
            #print(beam[0].parent, beam[0].everTreepath, beam[0].state)
            f.write(beam.getTreestr())
            f.write("\n")
        #exit(0)
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)
if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
        args.seed = sys.argv[2]
    else:
        test()
     #test()




