# MINLP.py

An other approach to the exhaustive search is to formulate an optimization problem, or in this case a, so called, Mixed-Integer Non-Linear Programming Problem.
The implementation is nothing more than the MCMF-problem extended with the capacity depending on the static positions of the jammers and users and the variable positions of the MANETNodes.

However, the current optimizer used (IPOPT) seems unable to solve it. Either, a different solver must be used or the formulation must be improved.

::: OptimumFinder.MINLP
    options:
      show_root_heading: false  # Avoid showing the module name as a header
