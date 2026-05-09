import math
import sys
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def optimalILP(costRank, K, item_to_group, group_to_item, ALPHA, BETA):
    m = gp.Model("fair-ra")
    m.params.OutputFlag = 0

    # Create variables
    assignmentVars = m.addMVar(shape=(DIM, DIM), vtype = GRB.BINARY)

    m.addConstr(assignmentVars @ np.ones(DIM) == np.ones(DIM))

    m.addConstr(assignmentVars.T @ np.ones(DIM) == np.ones(DIM))

    # Add fairness constraints
    fairness_ones = np.concatenate((np.ones(K), np.zeros(DIM - K)))
    for i in range(NUM_GROUPS):
        LB = math.floor(ALPHA[i] * K)
        UB = math.ceil(BETA[i] * K)
        m.addConstr(gp.quicksum([assignmentVars[j] @ fairness_ones for j in group_to_item[i]]) >= LB)
        m.addConstr(gp.quicksum([assignmentVars[j] @ fairness_ones for j in group_to_item[i]]) <= UB)

    # Create objective function
    m.setObjective(gp.quicksum([costRank[i] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)

    m.optimize()

    #Retrieve solution from the assignment variables
    output_rank = [0] * DIM
    #for i in range(DIM):
        #print(assignmentVars[i].X)
    vars = assignmentVars.X

    for i in range(DIM):
        for j in range(DIM):
            if vars[i][j] > 0.99:
                output_rank[j] = i
                break
            if vars[i][j] <= 0.99 and vars[i][j] >= 0.01:
                print("NOT INTEGRAL!!")

    return output_rank, m.objVal

def LP(rankings, costRank, K, item_to_group, group_to_item, ALPHA, BETA):
    m = gp.Model("lp")
    #m.params.OutputFlag = 0

    # Create variables
    assignmentVars = m.addMVar(shape=(DIM, DIM))

    m.addConstr(assignmentVars @ np.ones(DIM) == np.ones(DIM))

    assignmentVars_t = assignmentVars.T
    m.addConstr(assignmentVars_t @ np.ones(DIM) == np.ones(DIM))

    # Add fairness constraints
    fairness_ones = np.concatenate((np.ones(K), np.zeros(DIM - K)))
    for i in range(NUM_GROUPS):
        LB = math.floor(ALPHA[i] * K)
        UB = math.ceil(BETA[i] * K)
        m.addConstr(gp.quicksum([assignmentVars[j] @ fairness_ones for j in group_to_item[i]]) >= LB)
        m.addConstr(gp.quicksum([assignmentVars[j] @ fairness_ones for j in group_to_item[i]]) <= UB)

    # Create objective function
    m.setObjective(gp.quicksum([costRank[i] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)
    m.optimize()

    #Retrieve solution from the assignment variables
    output_rank = [0] * DIM
    for i in range(DIM):
        print(assignmentVars[i].X)
    vars = assignmentVars.X
    integral = 0
    for i in range(DIM):
        for j in range(DIM):
            if vars[i][j] > 0.99:
                output_rank[j] = i
            if vars[i][j] <= 0.99 and vars[i][j] >= 0.01:
                integral += 1
    print("Non-integral count", integral)
    return output_rank

def ourAlgo(costRank, K, LB, UB, item_to_group, group_to_item):

    objval = 0
    #solve the top K
    m = gp.Model("topk")
    m.params.OutputFlag = 0

    #use simplex method
    m.params.Method = 0

    # Create variables
    assignmentVars = m.addMVar(shape=(DIM, K))

    m.addConstr(assignmentVars @ np.ones(K) <= np.ones(DIM))

    #Each position needs at least one item 
    m.addConstr(assignmentVars.T @ np.ones(DIM) == np.ones(K))
    # Add fairness constraints
    
    for i in range(NUM_GROUPS):
        m.addConstr(gp.quicksum([assignmentVars[j] @ np.ones(K) for j in group_to_item[i]]) >= LB[i])
        m.addConstr(gp.quicksum([assignmentVars[j] @ np.ones(K) for j in group_to_item[i]]) <= UB[i])

    m.setObjective(gp.quicksum([costRank[i][:K] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)
    m.optimize()
    
    #Retrieve solution from the assignment variables
    output_rank = [0] * DIM
    #for i in range(DIM):
        #print(assignmentVars[i].X)
    assigned = set()

    assignvars = assignmentVars.X

    for i in range(DIM):
        for j in range(K):
            if assignvars[i][j] > 0.99:
                output_rank[j] = i
                assigned.add(i)
                break

    objval += m.ObjVal

    #Fill rest of ranking
    m = gp.Model("topk")
    m.params.OutputFlag = 0
    #use simplex method
    m.params.Method = 0

    assignmentVars = m.addMVar(shape=(DIM, DIM - K))

    isOne = np.ones(DIM)
    for i in assigned:
        isOne[i] = 0

    m.addConstr(assignmentVars @ np.ones(DIM - K) == isOne)

    m.addConstr(assignmentVars.T @ np.ones(DIM) == np.ones(DIM - K))
    m.setObjective(gp.quicksum([costRank[i][K:] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)
    m.optimize()

    assignvars = assignmentVars.X
    for i in range(DIM):
        for j in range(DIM - K):
            if assignvars[i][j] > 0.99:
                output_rank[j + K] = i
                assigned.add(i)
                break
    
    objval += m.ObjVal
    return output_rank, objval

def ourAlgoWrapper(costRank, K, item_to_group, group_to_item, ALPHA, BETA):
    #compute top K bounds
    LB = [math.floor(ALPHA[i] * K) for i in range(NUM_GROUPS)]
    UB = [math.ceil(BETA[i] * K) for i in range(NUM_GROUPS)]

    #print("Solving first step...")
    sigma1, sigma1val = ourAlgo(costRank, K, LB, UB, item_to_group, group_to_item)

    new_LB = [len(group_to_item[i]) - UB[i] for i in range(NUM_GROUPS)]
    new_UB = [len(group_to_item[i]) - LB[i] for i in range(NUM_GROUPS)]

    newCostRank = np.zeros((DIM, DIM), dtype = np.int16)
    for i in range(NUM_RANKINGS):
        ranking = rankings[i]
        for j in range(DIM):
            elem = ranking[j]
            for k in range(j+1, DIM):
                newCostRank[elem][k] += k - j
    newCostRank = np.flip(newCostRank, axis = 1)

    #print("Solving second step...")
    sigma2, sigma2val = ourAlgo(newCostRank, DIM - K, new_LB, new_UB, item_to_group, group_to_item)
    sigma2 = sigma2[::-1]

    return sigma1, sigma2
    
def bestFromInput(rankings, costRank, K, item_to_group, ALPHA, BETA):
    bestObj = 1e9
    bestRanking = None
    for i in range(len(rankings)):
        ranking = rankings[i]
        fair_ranking = getClosestRanking(ranking, item_to_group, ALPHA, BETA, K)
        objval = getObjCost(fair_ranking, costRank)
        if objval < bestObj:
            bestObj = objval
            bestRanking = fair_ranking
    return bestRanking, bestObj

def getClosestRanking(ranking, item_to_group, ALPHA, BETA, K):
    group_taken = []
    for i in range(NUM_GROUPS):
        group_taken.append([])

    LB = [math.floor(ALPHA[g] * K) for g in range(NUM_GROUPS)]
    UB = [math.ceil(BETA[g] * K) for g in range(NUM_GROUPS)]
    
    new_ranking = []
    total_taken = 0
    for i in range(DIM):
        item = ranking[i]
        group = item_to_group[item]
        if len(group_taken[group]) < LB[group]:
            group_taken[group].append(item)
            total_taken += 1

    for i in range(DIM):
        if total_taken >= K:
            break
        item = ranking[i]
        group = item_to_group[item]
        if item in group_taken[group]:
            continue
        if len(group_taken[group]) < UB[group]:
            group_taken[group].append(item)
            total_taken += 1

    new_ranking = []
    for i in range(DIM):
        item = ranking[i]
        group = item_to_group[item]
        if item in group_taken[group]:
            new_ranking.append(item)
    
    for i in range(DIM):
        item = ranking[i]
        group = item_to_group[item]
        if item not in group_taken[group]:
            new_ranking.append(item)
    
    return new_ranking

#helper function to return the weighted tournament corresponding to the rank aggregation problem
def Get_Frac_Tournament(rankings):
    element_count = len(rankings[0])
    frac_tournament = np.ndarray((element_count, element_count))
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = 0

    for ranking in rankings:
        for i in range(len(ranking)):
            for j in range(i+1, len(ranking)):
                frac_tournament[ranking[i]][ranking[j]] += 1
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = frac_tournament[i][j] / len(rankings)

    return frac_tournament
    
#helper function to recover ordering from acyclic tournament
def Topological_Sort(adj):
    n = len(adj)  
    in_degree = [0] * n

    for i in range(n):
        for j in range(n):
            if adj[i][j] > 0.5:
                in_degree[j] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_sort = []

    while queue:
        node = queue.popleft()
        topo_sort.append(node)

        for j in range(n):
            if adj[node][j] > 0.5:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return topo_sort
    
####
####
# End of helper functions
####

#This implementation of our algorithm uses KWIKSORT to solve the standard rank aggregation problem
#For details on KWIKSORT, see Ailon, Newman, Charikar 2007
def OurAlgo_KS(alphas, betas, rankings, id_attribute, num_attributes):
    element_count = len(rankings[0])

    #STEP 1: determining top-k elements
    #Construct weighted tournament, and then sort by indegrees, and take it as following the algorithm in the paper

    start_time = time.time()
    frac_tournament = Get_Frac_Tournament(rankings)

    fract_time = time.time()

    #List of lists.
    #List i contains tuples of elements with attribute i
    #tuple is in the form (element id, indegree)
    indegree_attr = []
    
    for attribute in range(num_attributes):
        indegree_attr.append([])
    for i in range(element_count):
        i_attr = id_attribute[i]
        indeg = 0
        for j in range(element_count):
            indeg += frac_tournament[j][i]
        indegree_attr[i_attr].append((i, indeg))
    for attr in range(num_attributes):
        indegree_attr[attr].sort(key = lambda ituple : ituple[1])
        
    topk_elements = set()
    elements_taken = [0] * num_attributes
    num_taken = 0
    #now, we get top k elements following the algo
    #take lower bound first
    #form combined list at same time
    indegree_combined = []
    for attr in range(num_attributes):
        for j in range(math.floor(alphas[attr] * K)):
            topk_elements.add(indegree_attr[attr][j][0])
            elements_taken[attr] += 1
        indegree_combined += indegree_attr[attr][math.floor(alphas[attr] * K):]
    
    #sort combined list, then take while respecting beta upper bounds
    indegree_combined.sort(key = lambda ituple : ituple[1])
    for i in range(len(indegree_combined)):
        if len(topk_elements) >= K:
            break
        element = indegree_combined[i]
        i_attr = id_attribute[element[0]]
        if elements_taken[i_attr] < math.ceil(betas[i_attr] * K):
            elements_taken[i_attr] += 1
            topk_elements.add(element[0])


    #STEP 2, we need to order the top-k.
    #Following the paper, we need to run rank aggregation over the two partitions.
    
    #In this implementation, Kwiksort is used to solve approximately, runs fast and easy to implement

    #left is top k, the front part
    rankings_left = []
    rankings_right = []
    for rank in rankings:
        left_rank = []
        right_rank = []
        for i in rank:
            if i in topk_elements:
                left_rank.append(i)
            else:
                right_rank.append(i)
        rankings_left.append(left_rank)
        rankings_right.append(right_rank)

    #NOTE: Because the elements of the reduced rankings are not a continuous 1 ... k, we need to relabel the elements to be 1 ... k, and save the mapping
    #so we can map the result back to these elements

    left_forward_map = {}
    left_backward_map = {}
    mapped_rankings_left = []
    for i in range(len(rankings_left[0])):
        left_forward_map[rankings_left[0][i]] = i
        left_backward_map[i] = rankings_left[0][i]
    mapped_rankings_left.append([i for i in range(len(rankings_left[0]))])
    for i in range(1, len(rankings_left)):
        mapped_rank = []
        for j in rankings_left[i]:
            mapped_rank.append(left_forward_map[j])
        mapped_rankings_left.append(mapped_rank)

    right_forward_map = {}
    right_backward_map = {}
    mapped_rankings_right = []
    for i in range(len(rankings_right[0])):
        right_forward_map[rankings_right[0][i]] = i
        right_backward_map[i] = rankings_right[0][i]
    mapped_rankings_right.append([i for i in range(len(rankings_right[0]))])
    for i in range(1, len(rankings_right)):
        mapped_rank = []
        for j in rankings_right[i]:
            mapped_rank.append(right_forward_map[j])
        mapped_rankings_right.append(mapped_rank)

    #Using the better of kwiksort and best from input algorithms from Ailon, Newman, Charikar 2007 paper
    leftKwiksort = Kwiksort(mapped_rankings_left)
    rightKwiksort = Kwiksort(mapped_rankings_right)
    leftInput = Best_From_Input(mapped_rankings_left)
    rightInput = Best_From_Input(mapped_rankings_right)

    if getKTObjective(leftKwiksort, mapped_rankings_left) < getKTObjective(leftInput, mapped_rankings_left):
        left_topo_sorted = leftKwiksort
    else:
        left_topo_sorted = leftInput

    if getKTObjective(rightKwiksort, mapped_rankings_right) < getKTObjective(rightInput, mapped_rankings_right):
        right_topo_sorted = rightKwiksort
    else:
        right_topo_sorted = rightInput
    
    #Re-map the topologically sorted elements, to the original elements using the backward maps

    left_original = [left_backward_map[i] for i in left_topo_sorted]
    right_original = [right_backward_map[i] for i in right_topo_sorted]

    output_ranking = left_original + right_original

    end_time = time.time()

    return output_ranking

##Helper functions for KWIKSORT
##Need to take bettter of best from input, and the kwiksort algorithm
def Best_From_Input(rankings):
    best_rank = []
    obj_value = 1e9
    for rank in rankings:
        median_cost = getKTObjective(rank, rankings)
        if median_cost < obj_value:
            obj_value = median_cost
            best_rank = rank
    return best_rank

def Kwiksort(rankings):
    frac_tournament = Get_Frac_Tournament(rankings)
    initial = [i for i in range(len(rankings[0]))]
    rank = DoKwiksort(initial, frac_tournament)
    return rank

def DoKwiksort(elements, frac_tournament):
    if len(elements) <= 1:
        return elements
    pivot = rng.choice(elements)

    left = []
    right = []
    for element in elements:
        if element != pivot:
            if frac_tournament[element][pivot] >= 0.5:
                left.append(element)
            else:
                right.append(element)
    return DoKwiksort(left, frac_tournament) + [pivot] + DoKwiksort(right, frac_tournament)

def getObjCost(sol, costRank):
    ccost = 0
    assigned = set()
    for i in range(len(sol)):
        ccost += int(costRank[sol[i]][i])
        assigned.add(sol[i])
    assert(len(assigned) == len(sol)), "Error: Output is not a ranking"
    return ccost

def isFairRanking(ranking, item_to_group, ALPHA, BETA, K):
    group_count = [0] * NUM_GROUPS
    for i in range(K):
        item = ranking[i]
        group = item_to_group[item]
        group_count[group] += 1
    for g in range(NUM_GROUPS):
        if group_count[g] < math.floor(ALPHA[g] * K) or group_count[g] > math.ceil(BETA[g] * K):
            return False
    return True

#Functions to compute kendall tau objective
def ktDist(first, second):
    mappedrank = []
    for i in range(len(second)):
        mappedrank.append(first.index(second[i]))
    cost, blank = mergesort(mappedrank)
    return cost

#mergesort to compute distance in nlogn time
#input: A single ranking
#output: Kendall tau distance to the ranking 1, 2, ..., n
def mergesort(ranking):
    if len(ranking) <= 1:
        return 0, ranking
    leftsum, leftrank = mergesort(ranking[:len(ranking)//2])
    rightsum, rightrank = mergesort(ranking[len(ranking)//2:])
    csum = leftsum + rightsum
    leftindex = 0
    rightindex = 0
    outrank = []
    while leftindex < len(leftrank) and rightindex < len(rightrank):
        if leftrank[leftindex] < rightrank[rightindex]:
            outrank.append(leftrank[leftindex])
            leftindex += 1
        else:
            outrank.append(rightrank[rightindex])
            csum += len(leftrank) - leftindex
            rightindex += 1
    if leftindex < len(leftrank):
        outrank += leftrank[leftindex:]
    if rightindex < len(rightrank):
        outrank += rightrank[rightindex:]
    return csum, outrank
    
def getKTObjective(query, rankings):
    median_cost = 0
    for rank in rankings:
        median_cost += ktDist(query, rank)
    return median_cost

for DIM in range(30, 60, 5):
    with open(r'football\week4.in', "r") as f:
        sys.stdin = f
        parameters = input().rstrip().split(" ")

        NUM_RANKINGS = int(parameters[0])
        NUM_GROUPS = int(parameters[2])

        ALPHA = []
        BETA = []
        rankings = []
        group_count = [0] * NUM_GROUPS

        for i in range(NUM_GROUPS):
            fairparams = input().rstrip().split(" ")
            ALPHA.append(float(fairparams[0]))
            BETA.append(float(fairparams[1]))

        for i in range(NUM_RANKINGS):
            ranking = input().rstrip().split(" ")
            ranking = [int(j) for j in ranking]
            ranking = [j for j in ranking if j < DIM]
            rankings.append(ranking)

        DIM = len(rankings[0])
        K = DIM//2

        #Read group mapping
        item_to_group = {}
        group_to_item = [[] for _ in range(NUM_GROUPS)]
        for i in range(DIM):
            inline = input().rstrip().split(" ")
            inline = [int(j) for j in inline]
            item_to_group[inline[0]] = inline[1]
            group_to_item[inline[1]].append(inline[0])
            group_count[inline[1]] += 1


        #This sets alpha/beta to proportion of input
        for g in range(NUM_GROUPS):
            ALPHA[g] = group_count[g] / DIM
            BETA[g] = group_count[g] / DIM

        #compute cost-rank matrix
        leftCostRank = np.zeros((DIM, DIM), dtype = np.int16)

        for i in range(NUM_RANKINGS):
            ranking = rankings[i]
            for j in range(DIM):
                elem = ranking[j]
                for k in range(j):
                    leftCostRank[elem][k] += abs(j - k)
        trueCostRank = np.zeros((DIM, DIM), dtype = np.int16)

        for i in range(NUM_RANKINGS):
            ranking = rankings[i]
            for j in range(DIM):
                elem = ranking[j]
                for k in range(DIM):
                    trueCostRank[elem][k] += abs(j - k)

        #print("ILP")
        sol1, objOPT = optimalILP(trueCostRank, K, item_to_group, group_to_item, ALPHA, BETA)
        #print(getKTObjective(sol1, rankings[:NUM_RANKINGS]))
        #print(sol1)
        #print(objOPT)
        #print("Time taken: ", time.time() - startTime)

        #print("Ours")
        #sol2 = LP(rankings, costRank, K, item_to_group, group_to_item, ALPHA, BETA)
        solsigma1, solsigma2 = ourAlgoWrapper(leftCostRank, K, item_to_group, group_to_item, ALPHA, BETA)

        #print(sol2)
        #print(objours)
        #print("Time taken: ", time.time() - startTime)
        #print(getKTObjective(sol2, rankings[:NUM_RANKINGS]))
        #print("BFI")
        sol3, obj3 = bestFromInput(rankings[:NUM_RANKINGS], trueCostRank, K, item_to_group, ALPHA, BETA)
        #obj3 =(getKTObjective(sol3, rankings[:NUM_RANKINGS]))

        #This code is to set seed for kwiksort, to allow reproducibility
        seed_val = 1
        for alpha in ALPHA:
            seed_val *= alpha * 100
        rng = np.random.default_rng([4, K, NUM_RANKINGS, DIM, int(seed_val)])
        sol4 = OurAlgo_KS(ALPHA, BETA, rankings[:NUM_RANKINGS], item_to_group, NUM_GROUPS)

        #print("KT")
        #print(obj4)
        objOPT = getObjCost(sol1, trueCostRank)
        objours = getObjCost(solsigma1, trueCostRank)
        objours2 = getObjCost(solsigma2, trueCostRank)
        obj3 = getObjCost(sol3, trueCostRank)
        obj4 = getObjCost(sol4, trueCostRank)
        '''objOPT = getKTObjective(sol1, rankings[:NUM_RANKINGS])
        objours = getKTObjective(solsigma1, rankings[:NUM_RANKINGS])
        obj3 = getKTObjective(sol3, rankings[:NUM_RANKINGS])
        obj4 = getKTObjective(sol4, rankings[:NUM_RANKINGS])'''
        print("week4", ",", NUM_RANKINGS, ",", K, ",", DIM, ",", objOPT, ",", objours, ",", objours2, ",", obj3, ",", obj4)
        