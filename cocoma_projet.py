import protocols as pt
from problem import Problem
import fairnessMeasures

import protocols

p2 = Problem(3,8,'borda',centralized=True)
#print(p2)

p1 = Problem(3,12,'borda',centralized=True)
print(p1)
pf = []
for u1 in p1.agent[1].u.values():
    for u2 in p1.agent[2].u.values():
        pf.append((u1,u2))

#sb=pt.singlesdoubles(p1)
print(" ***************** fin de signles doubles ******************")
sa = pt.originalSA(p1)
print(" ************* fin de original sequence ***************")
#sc=  pt.itersinglesdoubles(p1)

#print(" ************* fin de original itersingleDoubles ***************")


