# Spearman's Footrule Fair Rank Aggregation

# Files
The file `fair-ra.py` contains all four implemented algorithms for fair rank aggregation under Spearman's footrule distance.

The file `fair-ra-ktobj.py` contains our algorithm, KT and BFI for fair rank aggregation under Kendall's tau distance.

Football datasets are located in the folder `football`, with `week1.in` to `week16.in` as the 16 separate problem instances and movielens datasets are in `Movielens` as `movielens.in`.

The first line of all input files consists of three integers, which is the number of rankings ($n$), $k$, and the number of groups respectively. Following that are $n$ lines, each consisting of a ranking over the candidates 0...$d-1$. Finally, there are $d$ lines consisting of two integers. The first is the candidate, and the second is the group the candidate belongs to.

# Usage

You will need numpy, gurobipy, scipy and a gurobi license to run the programs; We use Gurobi as the LP solver for our algorithm implementation. See [their website](https://www.gurobi.com/academia/academic-program-and-licenses/) for more information on obtaining an academic license.

`requirements.txt` contains the Python packages used, including gurobipy, so installing those with pip and having a license should be sufficient to run the experiments.

`fair-ra.py` reads from standard input. For example, to run it with the Movielens dataset, use `python fair-ra.py < football\week1.in`.

The same holds for `fair-ra-ktobj.py`.

To adjust the parameters, i.e. $k, n, d$, one can directly manipulate the code to set the variables K, NUM_RANKINGS or DIM respectively (see in the code for examples), or change the input files.