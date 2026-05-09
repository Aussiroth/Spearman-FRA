# Spearman's Footrule Fair Rank Aggregation

# Files
The file `fair-ra.py` contains implementations for the optimal (ILP), our algorithm in the paper, and the 3-approximation from (Chakraborty et. al 2022) for fair rank aggregation under Spearman's footrule distance.

The file `fair-kt.py` contains the implementation of the fair rank aggregation under Kendall-tau distance used as the baseline KT. 

Football datasets are located in the folder `football`, with `week1.in` to `week16.in` as the 16 separate problem instances and movielens datasets are in `Movielens` as `movielens.in`.

The first line of all input files consists of three integers, which is the number of rankings ($n$), $k$, and the number of groups respectively. Following that are $n$ lines, each consisting of a ranking over the candidates 0...$d-1$. Finally, there are $d$ lines consisting of two integers. The first is the candidate, and the second is the group the candidate belongs to.

# Usage

You will need numpy, gurobipy, scipy and a gurobi license to run the programs; We use Gurobi as the LP solver for our algorithm implementation. See [their website](https://www.gurobi.com/academia/academic-program-and-licenses/) for more information on obtaining an academic license.

`requirements.txt` contains the Python packages used, including gurobipy, so installing those with pip and having a license should be sufficient to run the experiments.

`fair-ra.py` reads from standard input. For example, to run it with the Movielens dataset, use `python fair-ra.py < football\week1.in`.

The same holds for `fair-kt.py`.

To adjust the parameters, i.e. $k, n, d$, one can directly manipulate the code to set the variables K, NUM_RANKINGS or DIM respectively (see in the code for examples), or run some preprocessing to modify the input files.