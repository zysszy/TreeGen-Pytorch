import pickle
import numpy as np
rule = pickle.load(open('rulead.pkl', 'rb'))
print(len(rule))
res = []
rest = {}
for i in range(len(rule)):
    res.append(np.sum(rule[i]))
    if np.sum(rule[i]) not in rest:
        rest[np.sum(rule[i])] = 0
    rest[np.sum(rule[i])] += 1
print(rest)