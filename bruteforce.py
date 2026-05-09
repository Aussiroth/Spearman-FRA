from collections import deque
import math

def footrule_cost(first, second):
    cost = 0
    for i in range(len(first)):
        item = first[i]
        pos_second = second.index(item)
        cost += abs(i - pos_second)
    return cost

def get_obj_cost(rankings, sol):
    cost = 0
    for i in range(len(rankings)):
        cost += footrule_cost(sol, rankings[i])
    return cost

def is_fair_ranking(ranking, item_to_group, ALPHA, BETA):
    group_count = [0] * NUM_GROUPS
    for i in range(min(len(ranking), K)):
        group = item_to_group[ranking[i]]
        group_count[group] += 1
    for i in range(NUM_GROUPS):
        if group_count[i] < math.floor(ALPHA[i] * K) or group_count[i] > math.ceil(BETA[i] * K):
            return False
    return True

parameters = input().rstrip().split(" ")

NUM_RANKINGS = int(parameters[0])
K = int(parameters[1])
NUM_GROUPS = int(parameters[2])

ALPHA = []
BETA = []
rankings = []

for i in range(NUM_GROUPS):
    fairparams = input().rstrip().split(" ")
    ALPHA.append(float(fairparams[0]))
    BETA.append(float(fairparams[1]))

for i in range(NUM_RANKINGS):
    ranking = input().rstrip().split(" ")
    ranking = [int(j) for j in ranking]
    rankings.append(ranking)

DIM = len(rankings[0])

#Read group mapping
item_to_group = {}
group_to_item = [[] for _ in range(NUM_GROUPS)]
for i in range(DIM):
    inline = input().rstrip().split(" ")
    inline = [int(j) for j in inline]
    item_to_group[inline[0]] = inline[1]
    group_to_item[inline[1]].append(inline[0])

stack = deque()
stack.append([])
BEST = 1e9
bestsol = []
while len(stack) > 0:
    sol = stack.pop()
    if len(sol) >= K and is_fair_ranking(sol, item_to_group, ALPHA, BETA) == False:
        continue
    if len(sol) == DIM:
        obj_cost = get_obj_cost(rankings, sol)
        if obj_cost < BEST:
            BEST = obj_cost
            bestsol = [sol]
        elif obj_cost == BEST:
            bestsol.append(sol)
    else:
        for i in range(DIM):
            if i not in sol:
                new_sol = sol + [i]
                tmpcost = get_obj_cost(rankings, new_sol)
                if tmpcost <= BEST:
                    stack.append(new_sol)

print(BEST)
print(*bestsol)