# ExhaustiveSearch.py

To evaluate Agents and Baseline it is beneficial to know what the optimum throughput to be achieved in any scenario would be, regardless of the complexity of its calculations, since this is not a viable method consider to solve the MANET relocation problem. 

Hence, an exhaustive search algo was created which more or less efficiently searches the space with the MANET nodes to maximize throughput. It is based on some intuitive idea, which intuitively promises to find a solution very close to the optimum.

To use it one must create and [ExhausiveSearcher](#OptimumFinder.ExhaustiveSearch.ExhaustiveSearcher)-Object. The one can call [startSearch()](#OptimumFinder.ExhaustiveSearch.ExhaustiveSearcher.startSearch) and anything else will be taken care for. The idea behind making it a class is to simplify keeping the state of shared objects and the updating mechanism of the progress bar which is important. Otherwise, it is impossible to tell how long the search would take which can be very long.

__Intuitive Idea__
Since, in an exhaustive search one can never exactly reach a continuous space level search this is also not tried here. It is also important to know that even if one nodes moves by 1 in the playground of $[0,100]^2$ the change in throughput will be minimal. Hence a position allocation for the nodes that deviates from the optimum by $\epsilon \in [0,0.5)$ is acceptable. Hence, the idea is to do a first rough grid search and then a refined search, though limiting the space to search to only the sensible space, as follows:

1. Reduce the space to search to the convex hull of the users
2. Since, if there are jammers the optimum could be slightly outside of that hull extend it by a circle around the jammers position
3. Lay a rather rough grid over the working set (hull +- jammer circles)
4. Find all possible allocations on that set for the nodes
5. Find the top 5 allocations in terms of throughput
6. For those allocations refine the search and check again

::: OptimumFinder.ExhaustiveSearch
    options:
      show_root_heading: false  # Avoid showing the module name as a header