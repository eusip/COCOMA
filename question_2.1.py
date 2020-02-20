from problem import Problem
# import mipsolving
import fairnessMeasures as fairnessmeasures
# p1 = Problem(3,4,'uniform',centralized=True)
# print(p1)
# print(p1.printAllocation())

# p3 = Problem(4,6,'empty', centralized=True)
# p3.setUtilities([
#     {'r0':0,'r1':0,'r2':0,'r3':0,'r4':0,'r5':0},
#     {'r0':1,'r1':2,'r2':5,'r3':3,'r4':7,'r5':2},
#     {'r0':2,'r1':6,'r2':8,'r3':1,'r4':1,'r5':2},
#     {'r0':5,'r1':4,'r2':4,'r3':3,'r4':2,'r5':2}
# ])
# print (p3)
# print (p3.printAllocation())

# p2 = Problem(3,6,'normalized',centralized=False)
# print(p2)
# print(p2.printAllocation())
# print(fairnessmeasures.isProportional(p2))
# print(fairnessmeasures.egalitarianSW(p2))
#
# em = fairnessmeasures.envyMatrix(p2)
# print(em)
# print("There are ", fairnessmeasures.nbEnviousAgents(em), " envious agents")
# print("The maximum envy among two agents is ", fairnessmeasures.maxEnvy(em))

p = Problem(3, 4, 'empty', centralized=True)
p.setUtilities([
    {'r0': 6, 'r1': 6, 'r2': 1, 'r3': 7},
    {'r0': 1, 'r1': 3, 'r2': 9, 'r3': 7},
    {'r0': 8, 'r1': 0, 'r2': 4, 'r3': 8}
])

Prop_alloc = [[1,1,0,0], [0,0,1,0], [0,0,0,1]]
Envy_free_alloc = [[1,1,0,0], [0,0,1,0], [0,0,0,1]]
ESW_alloc = [[1,1,0,0], [0,0,1,0], [0,0,0,1]]
USW_alloc = [[1,1,0,0], [0,0,1,0], [0,0,0,1]]
Nash_alloc = [[0,1,0,1], [0,0,1,0], [1,0,0,0]]


