import os
from tqdm import tqdm
import pickle
import json
import numpy as np
lst = ["train", "dev", "test"]
rules = {"pad":0}
onelist =['body']
rulelist = []
fatherlist = []
fathername = []
depthlist = []
copynode = {}
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
def parseTree(treestr):
    tokens = treestr.split()
    root = Node("Module", 0)
    currnode = root
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = Node(x, i + 1)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    return root
maxnlnum = 40
hascopy = {}
def getcopyid(nls, name):
    global maxnlnum
    global hascopy
    lastcopyid = -1
    for i, x in enumerate(nls):
        if name.lower() == x.lower():
            lastcopyid = i
            if i not in hascopy:
                hascopy[i] = 1
                return i + 10000
    if lastcopyid != -1:
        return lastcopyid + 10000
    return -1
rulead = np.zeros([1772, 1772])
astnode = {"pad": 0, "Unknown": 1}
def getRule(node, nls, currId, d):
    global rules
    global onelist
    global rulelist
    global fatherlist
    global depthlist
    global copynode
    global rulead
    if node.name == "str_":
        assert(len(node.child) == 1)
    if len(node.child) == 0:
        return [], []
        if " -> End " not in rules:
            rules[" -> End "] = len(rules)
        return [rules[" -> End "]]
    child = node.child#sorted(node.child, key=lambda x:x.name)
    if len(node.child) == 1 and len(node.child[0].child) == 0:
        node.child[0].name = node.child[0].name.replace("!", "")
        copyid = getcopyid(nls, node.child[0].name)
    if len(node.child) == 1  and len(node.child[0].child) == 0 and copyid != -1:
        if len(node.child[0].child) != 0:
            print(node.child[0].name)
        copynode[node.name] = 1
        rulelist.append(copyid)
        fatherlist.append(currId)
#        rulead[rulelist[currId], 1771] = 1
#        rulead[1771, rulelist[currId]] = 1
        fathername.append(node.name)
        depthlist.append(d)
        currid = len(rulelist) - 1
        for x in child:
            getRule(x, nls, currId, d + 1)
            #rulelist.extend(a)
            #fatherlist.extend(b)
    else:
        if node.name not in onelist:
            rule = node.name + " -> "
            for x in child:
                rule += x.name + " "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
#            if currId != -1:
#                rulead[rulelist[currId], rulelist[-1]] = 1
#                rulead[rulelist[-1], rulelist[currId]] = 1
#            else:
#                rulead[770, rulelist[-1]] = 1
#                rulead[rulelist[-1], 770] = 1
            currid = len(rulelist) - 1
            for x in child:
                getRule(x, nls, currid, d + 1)
        else:
            #assert(0)
            for x in (child):
                rule = node.name + " -> " + x.name
                if rule in rules:
                    rulelist.append(rules[rule])
                else:
                    rules[rule] = len(rules)
                    rulelist.append(rules[rule])
#                rulead[rulelist[currId], rulelist[-1]] = 1
#                rulead[rulelist[-1], rulelist[currId]] = 1
                fatherlist.append(currId)
                fathername.append(node.name)
                depthlist.append(d)
                getRule(x, nls, len(rulelist) - 1, d + 1)
            rule = node.name + " -> End "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            rulead[rulelist[currId], rulelist[-1]] = 1
            rulead[rulelist[-1], rulelist[currId]] = 1
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
    '''if node.name == "root":
        print('rr')
        print('rr')
        print(rulelist)'''
    '''rule = " -> End "
    if rule in rules:
        rulelist.append(rules[rule])
    else:
        rules[rule] = len(rules)
        rulelist.append(rules[rule])'''
    #return rulelist, fatherlistd
def getTableName(f):
    global tablename
    lines = f.readlines()
    tabname = []
    dbid = ""
    tabname = []
    colnames = []
    for i in range(len(lines)):
        if i == 0:
            nl = lines[i].strip().split()
        if i == 1:
            originnl = lines[i].strip().split()
        if i == 2:
            dbid = lines[i].strip()
            for i, x in enumerate(tablename[dbid]['table_names_original']):
                tabname.append(x.lower())
                for j, y in enumerate(tablename[dbid]['column_names_original']):
                    if y[0] == i:
                        if y[1].lower() == "share":
                            y[1] = "share_"
                        colnames.append(y[1].lower())
    return nl, originnl, tabname, dbid, colnames
            
for x in lst:
    #inputdir = x + "_input/"
    #outputdir = x + "_output/"
    wf = open(x + "_process.txt", "w")
    f = open(x + ".txt", "r")
    lines = f.readlines()
    f.close()
    for i in tqdm(range(int(len(lines) / 2))):
        #fname = inputdir + str(i + 1) + ".txt"
        #ofname = outputdir + str(i + 1) + ".txt"
        nls = lines[2 * i].split("\t")#getTableName(f)
        asts = lines[2 * i + 1].strip()
        #wf.write(asts + "\n")
        hascopy = {}
        print(asts.split().count("^"))
        assert(len(asts.split()) == 2 * asts.split().count('^'))
        root = parseTree(asts)
        rulelist = []
        fatherlist = []
        fathername = []
        depthlist = []
        getRule(root, nls, -1, 2)
        wf.write(" ".join(nls))
        s = ""
        for x in rulelist:
            s += str(x) + " "
        wf.write(s + "\n")
        s = ""
        for x in fatherlist:
            s += str(x) + " "
        wf.write(s + "\n")
        s = ""
        for x in depthlist:
            s += str(x) + " "
        wf.write(s + "\n")
        wf.write(" ".join(fathername) + "\n")
        
        #print(rules)
        #print(asts)
wf.close()
wf = open("rule.pkl", "wb")
open("rulead.pkl", "wb").write(pickle.dumps(rulead))
#rules["start -> Module"] = len(rules)
#rules["start -> copyword"] = len(rules)
codead = np.zeros([565, 565])
for x in rules:
    lst = x.strip().lower().split()
    tmp = [lst[0]] + lst[2:]
    for y in tmp:
        if y not in astnode:
            astnode[y] = len(astnode)
    pid = astnode[lst[0]]
    for s in lst[2:]:
        tid = astnode[s]
#        codead[pid, tid] = 1
#        codead[tid, pid] = 1
open("Code_Voc.pkl", "wb").write(pickle.dumps(astnode))
open("codead.pkl", "wb").write(pickle.dumps(codead))
wf.write(pickle.dumps(rules))
wf.close()
print(rules)
print(astnode)
