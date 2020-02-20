# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:34:47 2016

@author: nicolas
"""
import heapq
import random
import fairnessMeasures
import numpy as np
import time
from problem import Problem
import matplotlib.pyplot as plt

###############################################################################
# Test for Single-Peaked Preference
###############################################################################
"""Implementation of the Single-Peaked Preference Algorithm detailed in V.Conitzer,
"Eliciting Single-Peaked Preferences Using Comparison Queries". 2009
Journal of Artificial Intelligence Research. """


def prefere(a1, a2):
    """
    'In this subsection, we focus on the setting where the elicitor knows the
    positions of the alternatives' (pg 168). In our case this implies resources
    are ordered from left to right r_1 to r_n.
    :param r1: ressource 1 e.g. 'r1'
    :param r2: resoource 2 e.g. 'r4'
    :return: True if r1 is preferred to r2; False otherwise
    """
    if a1 < a2:
        return True
    else:
        return False


def findPeak(u):
    """
    'The first algorithm serves to find the agent’s peak
    (most preferred alternative)' (pg. 169).
    :param agent: dictionary of agent preferences over all objects
    :return: find the single-peak of an agent given positions
    """

    l = 0
    r = len(u) - 1
    while l < r:
        m1 = int((l + r) / 2)
        m2 = m1 + 1
        # print("m1 has the position", m1, "in the list of alternatives.",
        #       "m2 has the position", m2, "in the list of alternatives.")
        if prefere(list(u.values())[m1], list(u.values())[m2]):
            r = m1
        else:
            l = m2
    print("m1 has the position", m1, "in the list of alternatives.",
          "m2 has the position", m2, "in the list of alternatives.")
    print("The most preferred item is: ", list(u.keys())[m1])
    return l


def singlePeaked(agent):
    """
    'Once we have found the peak, we can continue to construct the 
    agent’s ranking of the alternatives as follows[...]
    Once we have determined the ranking of either the leftmost 
    or the rightmost alternative, we can construct the remainder of
    the ranking without asking any more queries (by simply 
    ranking the remaining alternatives according to proximity to 
    the peak' (pg. 169-170).
    :param agent: dictionary of agent preferences over all objects
    :return: find the single-peak of an agent given positions
    """
    u = {}
    for i in range(len(agent.hold)):
        u[agent.hold[i]] = agent.u.get(agent.hold[i])
    t = findPeak(u)
    s = list(u.values())[t]
    print("position of peak = %d and its utility = %d" % (t, s))
    l = t - 1
    r = t + 1
    c = list()
    c.append(list(u.keys())[t])
    res = []
    res.append(s)
    while l >= 0 and r <= (len(u) - 1):
        if prefere(list(u.values())[l], list(u.values())[r]):
            res.append(list(u.values())[l])
            c.append(list(u.keys())[l])
            l -= 1
        else:
            res.append(list(u.values())[r])
            c.append(list(u.keys())[r])
            r += 1
    while l >= 0:
        res.append(list(u.values())[l])
        c.append(list(u.keys())[l])
        l -= 1
    while r <= len(u) - 1:
        res.append(list(u.values())[r])
        c.append(list(u.keys())[r])

        r += 1
    print("*** ressources single peaked = ", c)
    #plt.plot(res), '*-'
    #plt.show()
    return c


###############################################################################
# New Functions for originalSA(p), restrictedSA(p),
# singlesdoubles(p) and itersinglesdoubles(p)
###############################################################################


def allocate(p, agent, item):
    '''this function allocations a specified item from the centralized agent to any other agent'''
    if list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(item)] in p.agent[0].hold:
        p.agent[0].giveItem(list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(item)])
        p.agent[agent].getItem(list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(item)])
    else:
        print("What are you doing? This item cannot be allocated as it has already been allocated.")


def find_best(p, agent):
    '''this function takes the agent as an input and checks for the most
    preferred available object in p.agent[0].hold'''
    pref = 1
    while pref <= p.m and len(p.agent[0].hold) != 0:
        if pref < p.m:
            if list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(pref)] \
                    in p.agent[0].hold:
                return list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(pref)]
            else:
                pref += 1

def find_sb(p, agent):
    """this function takes the agent as an input and checks for the second most
    preferred available object in p.agent[0].hold"""
    pref = 1
    while pref < p.m and len(p.agent[0].hold) >= 2:
        if list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(pref+1)] \
                in p.agent[0].hold:
            return list(p.agent[agent].u.keys())[list(p.agent[agent].u.values()).index(pref+1)]
        else:
            pref += 1


def find_max_min(p, unallocated):
    """this function calculates the max-min rank of a two-agent problem"""
    a1 = []
    a2 = []
    '''this function determines the max-min rank for two agents'''
    for r in unallocated:
        a1.append(list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(r))])
        a2.append(list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(r))])
    # print('a1: ', a1 ,'a2: ', a2)
    k = np.max(np.minimum(a1, a2))
    return k


def envy_free_chk(p, agent, own_r, adv_r, own_chk, adv_chk):
    """This function checks if a proposed allocation will be envy-free."""
    own_tot = 0
    adv_tot = 0
    own_r = own_r.copy()
    if own_chk is not None:
        own_r.append(own_chk)
    adv_r = adv_r.copy()
    if adv_chk is not None:
        adv_r.append(adv_chk)
    for item in own_r:
        own_tot = list(p.agent[agent].u.values())[list(p.agent[agent].u.keys()).index(str(item))]
        own_tot += own_tot
    for item in adv_r:
        adv_tot = list(p.agent[agent].u.values())[list(p.agent[agent].u.keys()).index(str(item))]
        adv_tot += adv_tot
    if own_tot - adv_tot < 0:
        return False, print("There is envy in this allocation!")  # print("own_r contains: ", own_r, "adv_r: ", adv_r)
    else:
        return True, print("This is an envy-free allocation.")


###############################################################################
# Original Sequential Algorithm
###############################################################################


def originalSA(p):
    '''
    runs the original Sequential Algorithm on problem p
    to be used with centralized problem
    and only two agents
    '''
    start = time.time()
    if p.n != 3:
        print("Warning: This algorithm is intended for only two agents.")
        print("Note: Only the two first agents will be considered.")
    l = 1
    while (l <= p.m) and len(p.agent[0].hold) > 0:
        print("l:", l)
        print("Unallocated: ", p.agent[0].hold)
        s1_k = []
        s2_k = []
        for pref in range(1, l+1):
            if list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)] in p.agent[0].hold:
                s1_k.append(list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)])
            if list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)] in p.agent[0].hold:
                s2_k.append(list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)])
        print('Agent 1 item preference for this round: ', s1_k,
              'Agent 2 item preference for this round: ', s2_k)
        for i in range(len(s1_k)):
            for j in range(len(s2_k)):
                if s1_k[i] != s2_k[j]:
                    allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).
                             index(str(s1_k[i]))])
                    allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).
                             index(str(s2_k[j]))])
                    print("Item", s1_k[i], "has been allocated to Agent 1.",
                          "Item", s2_k[j], "has been allocated to Agent 2.")
                    break
            else:
                print("No allocation has been found. Recalculating l.")
                continue
            break
        # exit()
        print("The allocation for round", l, "is as follows: Unallocated: ", p.agent[0].hold,
              "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
        l += 1
    print("The final allocation is as follows: Unallocated: ", p.agent[0].hold,
          "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
    print("\n   single-Peaked for agent 1")
    singlePeaked(p.agent[1])
    print("\n   single-Peaked for agent 2 ")
    singlePeaked(p.agent[2])
    if len(p.agent[0].hold) != 0:
        print("There is NO envy-free full allocation of items.")
    end = time.time()
    print("Execution time: ", end - start)


###############################################################################
# Restricted Sequential Algorithm
###############################################################################


def restrictedSA(p):
    '''
    runs the restricted Sequential Algorithm on problem p
    to be used with centralized problem and only two agents
    '''
    start = time.time()
    if p.n != 3:
        print("Warning: This algorithm is intended for only two agents.")
        print("Note: Only the two first agents will be considered.")
    l = 1
    while (l <= p.m) and len(p.agent[0].hold) > 0:
        print("l:", l)
        print("Unallocated: ", p.agent[0].hold)
        s1_k = []
        s2_k = []
        for pref in range(1, l+1):
            if list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)] in p.agent[0].hold:
                s1_k.append(list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)])
            if list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)] in p.agent[0].hold:
                s2_k.append(list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)])
        print('Agent 1 item preference for this round: ', s1_k,
              'Agent 2 item preference for this round: ', s2_k)
        top1 = find_best(p, 1)
        top2 = find_best(p, 2)
        if top1 != top2:
            allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))])
            allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(top2))])
            print("l:", l, "Unallocated: ", p.agent[0].hold, "Agent 1: ", p.agent[1].hold,
                  "Agent 2: ", p.agent[2].hold, ".")
            if top1 in p.agent[1].hold and len(s1_k) != 0:
                s1_k.remove((str(top1)))
            if top2 in p.agent[2].hold and len(s2_k) != 0:
                s2_k.remove((str(top2)))
        else:
            if len(s1_k) > 1 and len(s2_k) > 0 and s1_k[1] in p.agent[0].hold and s2_k[0] in p.agent[0].hold:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).
                         index(str(s1_k[1]))])
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).
                         index(str(s2_k[0]))])
            else:
                print("Only one item in H_a(l). Now checking H_b(l).")
            if len(s2_k) > 1 and len(s1_k) > 0 and s1_k[0] in p.agent[0].hold and s2_k[1] in p.agent[0].hold:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).
                         index(str(s1_k[0]))])
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).
                         index(str(s2_k[1]))])
            else:
                print("Only one item in H_b(l). Recalculating l.")
        print("The allocation for round", l, "is as follows: Unallocated: ", p.agent[0].hold,
              "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
        l += 1
    print("The final allocation is as follows: Unallocated: ", p.agent[0].hold,
          "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
    print("\n   single-Peaked for agent 1")
    singlePeaked(p.agent[1])
    print("\n   single-Peaked for agent 2 ")
    singlePeaked(p.agent[2])
    if len(p.agent[0].hold) != 0:
        print("There is NO envy-free full allocation of items.")
    end = time.time()
    print("Execution time: ", end - start)

###############################################################################
# Singles-Doubles Algorithm
###############################################################################


def singlesdoubles(p):
    """this function runs the original Sequential Algorithm on problem p;
    it is designed to be used on a centralized problem with only two agents
    """
    start = time.time()
    rnd = 1
    while len(p.agent[0].hold) > 1 and rnd < 50:  # rnd < 50 just to prevent an infinite loop
        k = find_max_min(p, p.agent[0].hold)
        # print('Currently unallocated: ', p.agent[0].hold)
        print('k: ', k)
        s1_k = []
        s2_k = []
        last_agent_0_hold = p.agent[0].hold.copy()
        for pref in range(1, k+1):
            if list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)] in p.agent[0].hold:
                s1_k.append(list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)])
            if list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)] in p.agent[0].hold:
                s2_k.append(list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)])
        print('Agent 1 ordered preference for this round: ', s1_k,
              'Agent 2 ordered preference for this round: ', s2_k)
        '''phase 1 allocation'''
        print("Phase 1 - beginning allocation attempt.")
        for i in s1_k:
            if i not in s2_k:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(i))])
        for i in s2_k:
            if i not in s1_k:
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(i))])
        db = list(set(s1_k).intersection(s2_k))
        print("Max-min rank (k):", k, "Unallocated: ", p.agent[0].hold, "Agent 1: ", p.agent[1].hold,
              "Agent 2: ", p.agent[2].hold, ". Doubles: ", db)
        print("Phase 1 - allocation complete.")
        # exit()
        '''Phase 2 allocation'''
        print("Phase 2 Doubles - beginning 1st allocation attempt.")
        top1 = find_best(p, 1)
        top2 = find_best(p, 2)
        if top1 != top2:
            allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))])
            allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(top2))])
            print("Max-min rank (k): ", k, "Unallocated: ", p.agent[0].hold, "Agent 1: ",
                  p.agent[1].hold, "Agent 2: ", p.agent[2].hold, ".")
        else:
            print("No luck. Next phase 2 attempt.")
        # exit()
        print("Phase 2 Doubles - beginning 2nd allocation attempt.")
        top1 = find_best(p, 1)
        sb2 = find_sb(p, 2)
        # print('top1: ', top1, 'sb2: ', sb2)
        if top1 != sb2:
            '''Check if the proposed allocation is envy free'''
            if sb2 is not None:
                res1 = envy_free_chk(p, 1, p.agent[1].hold, p.agent[2].hold, str(top1), str(sb2))
                res2 = envy_free_chk(p, 2, p.agent[2].hold, p.agent[1].hold, str(sb2), str(top1))
            else:
                res1 = [False]
                res2 = [False]
            # print('res1: ', res1[0], 'res2: ', res2[0])
            if res1[0] is True and res2[0] is True:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))])
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(sb2))])
                print("1: ", list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))],
                      "2: ", list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(sb2))])
                print("Max-min rank (k):", k, "Unallocated: ", p.agent[0].hold, "Agent 1: ",
                      p.agent[1].hold, "Agent 2: ", p.agent[2].hold, ".")
            else:
                print("This is not an envy-free allocation. Moving to the next pair of "
                      "most preferred items.")
        else:
            print("No luck. Next final phase attempt.")
        # exit()
        print("Phase 2 Doubles - beginning 3rd allocation attempt.")
        top2 = find_best(p, 2)
        sb1 = find_sb(p, 1)
        # print('top1: ', top1, 'sb2: ', sb2)
        if top2 != sb1:
            '''Check if the proposed allocation is envy free'''
            if sb1 is not None:
                res1 = envy_free_chk(p, 2, p.agent[2].hold, p.agent[1].hold, str(top2), str(sb1))
                res2 = envy_free_chk(p, 1, p.agent[1].hold, p.agent[2].hold, str(sb1), str(top2))
            else:
                res1 = [False]
                res2 = [False]
            # print('res1: ', res1[0], 'res2: ', res2[0])
            if res1[0] is True and res2[0] is True:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top2))])
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(sb1))])
            else:
                print("This is not an envy-free allocation. Recalculating k.")
        print("The allocation for round", rnd, "is as follows: Unallocated: ", p.agent[0].hold,
              "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
        if last_agent_0_hold == p.agent[0].hold:
            break
        rnd += 1
    print("The final allocation is as follows: Unallocated: ",  p.agent[0].hold,
          "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
    print("\n   single-Peaked for agent 1")
    singlePeaked(p.agent[1])
    print("\n   single-Peaked for agent 2 ")
    singlePeaked(p.agent[2])
    if len(p.agent[0].hold) != 0:
        print("There is NO envy-free full allocation of items.")
    else:
        print("This is an envy-free full allocation of items.")
    end = time.time()
    print("Execution time: ", end-start)


###############################################################################
# Iterated Singles-Doubles Algorithm
###############################################################################


def itersinglesdoubles(p):
    """this function runs the original Sequential Algorithm on problem p;
    it is designed to be used on a centralized problem with only two agents
    """
    start = time.time()
    rnd = 1
    while len(p.agent[0].hold) > 1 and rnd < 50:  # rnd < 50 just to prevent an infinite loop
        last_agent_0_hold = p.agent[0].hold.copy()
        inner_rnd = 1
        while len(p.agent[0].hold) > 0 and inner_rnd < 25:  # inner_rnd < 20 just to prevent
                                                            # an infinite loop
            k = find_max_min(p, p.agent[0].hold)
            # print('Currently unallocated: ', p.agent[0].hold)
            print('k: ', k)
            s1_k = []
            s2_k = []
            last_singles = p.agent[0].hold.copy()
            for pref in range(1, k+1):
                if list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)] in p.agent[0].hold:
                    s1_k.append(list(p.agent[1].u.keys())[list(p.agent[1].u.values()).index(pref)])
                if list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)] in p.agent[0].hold:
                    s2_k.append(list(p.agent[2].u.keys())[list(p.agent[2].u.values()).index(pref)])
            print('Agent 1 ordered preference for this round: ', s1_k,
                  'Agent 2 ordered preference for this round: ', s2_k)
            '''phase 1 allocation'''
            print("Phase 1 - beginning allocation attempt.")
            for i in s1_k:
                if i not in s2_k:
                    allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(i))])
            for i in s2_k:
                if i not in s1_k:
                    allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(i))])
            db = list(set(s1_k).intersection(s2_k))
            print("Max-min rank (k):", k, "Unallocated: ", p.agent[0].hold, "Agent 1: ", p.agent[1].hold,
                  "Agent 2: ", p.agent[2].hold, ". Doubles: ", db)
            if last_singles == p.agent[0].hold:
                print("Phase 1 - allocation complete.")
                break
        print("The final phase 1 allocation is as follows: Unallocated: ", p.agent[0].hold,
              "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
        # exit()
        '''Phase 2 allocation'''
        print("Phase 2 Doubles - beginning 1st allocation attempt.")
        top1 = find_best(p, 1)
        top2 = find_best(p, 2)
        if top1 != top2:
            allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))])
            allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(top2))])
            print("Max-min rank (k): ", k, "Unallocated: ", p.agent[0].hold, "Agent 1: ",
                  p.agent[1].hold, "Agent 2: ", p.agent[2].hold, ".")
        else:
            print("No luck. Next phase 2 attempt.")
        # exit()
        print("Phase 2 Doubles - beginning 2nd allocation attempt.")
        top1 = find_best(p, 1)
        sb2 = find_sb(p, 2)
        # print('top1: ', top1, 'sb2: ', sb2)
        if top1 != sb2:
            '''Check if the proposed allocation is envy free'''
            if sb2 is not None:
                res1 = envy_free_chk(p, 1, p.agent[1].hold, p.agent[2].hold, str(top1), str(sb2))
                res2 = envy_free_chk(p, 2, p.agent[2].hold, p.agent[1].hold, str(sb2), str(top1))
            else:
                res1 = [False]
                res2 = [False]
            # print('res1: ', res1[0], 'res2: ', res2[0])
            if res1[0] is True and res2[0] is True:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))])
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(sb2))])
                print("1: ", list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top1))],
                      "2: ", list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(sb2))])
                print("Max-min rank (k):", k, "Unallocated: ", p.agent[0].hold, "Agent 1: ",
                      p.agent[1].hold, "Agent 2: ", p.agent[2].hold, ".")
            else:
                print("This is not an envy-free allocation. Moving to the next pair of "
                      "most preferred items.")
        else:
            print("No luck. Next final phase attempt.")
        # exit()
        print("Phase 2 Doubles - beginning 3rd allocation attempt.")
        top2 = find_best(p, 2)
        sb1 = find_sb(p, 1)
        # print('top1: ', top1, 'sb2: ', sb2)
        if top2 != sb1:
            '''Check if the proposed allocation is envy free'''
            if sb1 is not None:
                res1 = envy_free_chk(p, 2, p.agent[2].hold, p.agent[1].hold, str(top2), str(sb1))
                res2 = envy_free_chk(p, 1, p.agent[1].hold, p.agent[2].hold, str(sb1), str(top2))
            else:
                res1 = [False]
                res2 = [False]
            print('res1: ', res1[0], 'res2: ', res2[0])
            if res1[0] is True and res2[0] is True:
                allocate(p, 1, list(p.agent[1].u.values())[list(p.agent[1].u.keys()).index(str(top2))])
                allocate(p, 2, list(p.agent[2].u.values())[list(p.agent[2].u.keys()).index(str(sb1))])
            else:
                print("This is not an envy-free allocation. Recalculating k.")
        print("The allocation for round", rnd, "is as follows: Unallocated: ", p.agent[0].hold,
              "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
        if last_agent_0_hold == p.agent[0].hold:
            break
        rnd += 1
    print("The final allocation is as follows: Unallocated: ",  p.agent[0].hold,
          "Agent 1: ", p.agent[1].hold, "Agent 2: ", p.agent[2].hold)
    print("\n   single-Peaked for agent 1")
    singlePeaked(p.agent[1])
    print("\n   single-Peaked for agent 2 ")
    singlePeaked(p.agent[2])
    if len(p.agent[0].hold) != 0:
        print("There is NO envy-free full allocation of items.")
    else:
        print("This is an envy-free full allocation of items.")
    end = time.time()
    print("Execution time: ", end-start)


###############################################################################
# Adjusted Winner Procedure
###############################################################################

def adjustedWinner(p,verbose=True):
    '''
    runs the adjusted winner on problem p
    to be used with centralized problem
    and only two agents
    '''
    if p.n!=3:
        print("Warning: Adjusted Winner must be used with two agents.")
        print("Note: Only the two first agents will be considered.")
    # the allocation phase
    for r in p.agent[0].hold:   
        if p.agent[1].u[r]>p.agent[2].u[r]:
            p.agent[1].getItem(r)
        else:
            p.agent[2].getItem(r)     
    if verbose:
        print ("Allocation phase:")
        print (p.printAllocation())
    # happiest / saddest agent
    if p.agent[1].current_u>p.agent[2].current_u:
        high, low = 1,2
    else:
        high, low = 2,1
    # ranking the resources (of the rich)
    h = [] # using a heapqueue with u_h/u_l as comparison value
    for r in p.agent[high].hold:  
        ratio = p.agent[high].u[r] / p.agent[low].u[r]
        ratio = round(ratio,3) # to use the float as a dict key
        heapq.heappush(h,(ratio,r))
    print (h)
    # now inspect resources by priority oder
    while p.agent[high].current_u>p.agent[low].current_u:
        _,r = heapq.heappop(h)
        p.agent[low].getItem(r)
        p.agent[high].giveItem(r)
        if verbose:
            print ("Resource ",r , " moves from ", high, " to ", low)
    if p.agent[1].current_u != p.agent[2].current_u:
        part_of_low = (p.agent[low].current_u - p.agent[high].current_u) / \
        (p.agent[high].u[r]+p.agent[low].u[r])
        if verbose:
            print ("Resource ", r, " will be splitted!")
            print ("Agent ", low, " gets ", round(part_of_low,3), " of resource ", r)
    return


###############################################################################
# Picking Sequences
###############################################################################


def generateSequence(n,m,type_sequence):
    '''
    @n: number of agents
    @m: number of resources
    @t: type of sequence (balanced, alternate; etc.)
    '''
    if (m%n)!=0:
        print("Warning: number of resources not divisible by number of agents")
    sequence = []
    if type_sequence == "repeated":
        pass
    if type_sequence == "balanced":
        s = list(range(1,n+1))
        s_inv = s[::-1]
        
        for i in range(int(m/(2*n))):
            sequence += s + s_inv         
        
    return sequence

def pickingSequence(p,sequence,verbose=False):
    '''
    given a problem p and a sequence s (of integers)
    simulates agents picking at their turn from agent 0
    (supposed to be used with auctionneer problem)
    '''
    if len(sequence)!=p.m:
        print("The sequence length is different from the number of resources!")
    for i in sequence:
        best_utility = -1
        best_resource_to_pick=""
        for r in p.agent[0].hold:
            if p.agent[i].u[r]>best_utility:
                best_utility = p.agent[i].u[r]
                best_resource_to_pick = r
        if verbose==True:
            print ("agent ", i, " picks ", best_resource_to_pick)
        p.agent[i].getItem(best_resource_to_pick)
        p.agent[0].giveItem(best_resource_to_pick)
    return
   

###############################################################################
# Lipton et al.
###############################################################################

def lipton(p,verbose=True):
    '''
    runs the Lipton et al. protocol
    '''
    m = fairnessMeasures.envyMatrix(p)
    for j,r in enumerate(p.resources): # we allocate all resources one by one
        g = fairnessMeasures.buildEnvyGraph(m)
        if verbose:
            print(p.printAllocation())
            print("envy graph:", g)
            print ("allocating resource ", r)
        
        m = fairnessMeasures.envyMatrix(p)
        #g = fairnessMeasures.buildEnvyGraph(m)
        _,c = fairnessMeasures.checkCycle(g) # a cycle must exist
        while (c!=[]): # stop when acyclic graph
            if verbose:
                print("solving the cycle:",c)
            p.cycleReallocation(c)
            m = fairnessMeasures.envyMatrix(p)
            g = fairnessMeasures.buildEnvyGraph(m)
            _,c = fairnessMeasures.checkCycle(g) 
        
        for i in range(1,p.n):
            if not(fairnessMeasures.envied(m,i)):
                p.agent[i].getItem(r)
                p.agent[0].giveItem(r)
                break        
         
            #print(p.printAllocation())
        m = fairnessMeasures.envyMatrix(p)
    if verbose:
        g = fairnessMeasures.buildEnvyGraph(m)
        print("envy graph:", g)
    return


###############################################################################
# Local Exchanges
###############################################################################

def rationalSwapDeal(p,x,y,verbose=True):
    '''
    checks if there are rational 1-deal between agents x and y
    and performs all of them if possible (no further heuristic for choice)
    '''
    deal = False
    for rx in p.agent[x].hold: 
        for ry in p.agent[y].hold:
            if p.agent[x].u[rx]<p.agent[x].u[ry] and p.agent[y].u[rx]>p.agent[y].u[ry]:
                if verbose == True:                 
                    print ("deal between ", x, " and ", y, "for ", rx, " and ", ry)
                p.agent[x].getItem(ry)
                p.agent[x].giveItem(rx)
                p.agent[y].getItem(rx)
                p.agent[y].giveItem(ry)
                deal = True
                break
    return deal
                
                
def randomDynamics(p):
    testedPairs = []
    allPairs = [(x,y) for x in range(p.n) for y in range(p.n)]
    #random.shuffle(allPairs)
    
    while len(testedPairs) != len(allPairs): 
        candidatePairs = [(x,y) for (x,y) in allPairs if (x,y) not in testedPairs]   
        #print (candidatePairs)
        # choice in all pairs - tested
        (x,y) = random.choice(candidatePairs)    

        if not(rationalSwapDeal(p,x,y)):
            testedPairs += (x,y)
        else:
            testedPairs = []
            print (p.printAllocation())
    return



    
    
    
    


