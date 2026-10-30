# Network.py
Models the network. Here, `Users` only send data over `MANETNodes` to limit interference. The network is modeled as a directed graph using [NetworkX](https://networkx.org/). Where each edge represents a possible communication route. 

__Routing__
Routing is done by considering all Users send at once. There are different routing algorithms that were implemented. Generally, the MCCMF formulation is the aim to use, to simulate perfect routing. This unfortunately can be slow in execution, thus mostly the sequential max flow algorithm --- and approximation of the MCCMF problem --- is used.

::: MANET.Network
    options:
      show_root_heading: false  # Avoid showing the module name as a header
