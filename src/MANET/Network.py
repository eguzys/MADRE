# Type hinting
from typing import Callable

# Matrix math
import numpy as np
from scipy.sparse import csr_array

# Graph network modelling
import networkx as nx

# Own MANET classes
from .Components import MANETNode, Jammer, User, NetworkComponent
from .Routing import MaxFlow

"""
# Network.py
Models the network. Here, `Users` only send data over `MANETNodes` to limit interference. The network is modeled as a directed graph using [NetworkX](https://networkx.org/). Where each edge represents a possible communication route. 

__Routing__
Routing is done by considering all Users send at once. There are different routing algorithms that were implemented. Generally, the MCCMF formulation is the aim to use, to simulate perfect routing. This unfortunately can be slow in execution, thus mostly the sequential max flow algorithm --- and approximation of the MCCMF problem --- is used.
"""


class Network:
    """
    Models the network environment for the simulation.

    Attributes:
        nodes (np.ndarray): Array of MANET nodes in the network.
        jammers (np.ndarray): Array of jammers in the network.
        users (np.ndarray): Array of users in the network.
        graph (nx.DiGraph): Directed graph representing the network.
        np_random (np.random.Generator): Random number generator.
        networkThroughput (float): Total throughput of the network.
        capacity_matrix (np.ndarray): Matrix representing the capacities between network components.
        routing_func (Callable): Function determining how offered load is routed
    """

    def __init__(
        self,
        nodes: np.ndarray,
        jammers: np.ndarray,
        users: np.ndarray,
        np_random: np.random.Generator = np.random,
    ) -> None:
        """
        Initializes the network with nodes, jammers, users, and edges.

        Args:
            nodes (np.ndarray): Array of MANET nodes in the network.
            jammers (np.ndarray): Array of jammers in the network.
            users (np.ndarray): Array of users in the network.
            np_random (np.random.Generator, optional): Random number generator (default is np.random).

        Returns:
            None
        """
        self.nodes = nodes
        self.jammers = jammers
        self.users = users

        self.graph = nx.DiGraph()
        self.initializeGraph()

        # User random generator from environment if possible
        self.np_random = np_random

        self.networkThroughput = 0

        self.throughput_per_drone: dict = {}  # {node.id: {"in": float, "out": float}}, per step
        self.per_commodity_flow = {}  # {source_user: {node_u: {node_v: flow}}}, per step

        # Initialize capacity matrix
        self.capacity_matrix = None
        self.updateCapacityMatrix()

        # Routing function
        self.routing_func = MaxFlow.sequential_max_flow

    def initializeGraph(self) -> None:
        """
        Initializes the network graph with nodes and edges.

        Returns:
            None
        """
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_nodes_from(self.users)

    def updateNetwork(self) -> None:
        """
        Updates the network state.

        Returns:
            None
        """
        self.resetThroughput()
        self.resetValues()

        self.updateJammersEffect()
        self.updateSignals()
        self.updateEdges()
        self.updateCapacityMatrix()

    def resetThroughput(self) -> None:
        """
        Resets the throughput of all network components.

        Returns:
            None
        """
        self.networkThroughput = 0
        self.throughput_per_drone = {}
        self.per_commodity_flow = {}

    def resetValues(self) -> None:
        """
        Resets all values, which all network components receive.

        Returns:
            None
        """
        for node in self.nodes:
            node: MANETNode
            node.resetAll()

        for user in self.users:
            user: User
            user.resetAll()

        for jammer in self.jammers:
            jammer: Jammer
            jammer.resetAll()

    def updateSignals(self) -> None:
        """
        Updates the signals between network components.

        Returns:
            None
        """
        for node in self.nodes:
            node: MANETNode
            node.transmitSignal(self.nodes)
            node.transmitSignal(self.users)
            node.transmitSignal(self.jammers)

        for user in self.users:
            user: User
            user.transmitSignal(self.nodes)
            user.transmitSignal(self.jammers)

    def updateJammersEffect(self) -> None:
        """
        Updates the effect of jammers on the network components.

        Returns:
            None
        """
        for jammer in self.jammers:
            jammer: Jammer
            jammer.jamComps(self.nodes)
            jammer.jamComps(self.users)

    def updateEdges(self) -> None:
        """
        Updates the edges between network components.

        Returns:
            None
        """
        self.graph.clear_edges()
        self.addEdges()
        self.updateCapacityMatrix()

    def addEdges(self) -> None:
        """
        Adds edges between network components in the graph.

        Returns:
            None
        """
        self._addNodeEdges()
        self._addUserEdges()

    def _addNodeEdges(self) -> None:
        """
        Adds edges between nodes and other nodes/users.

        Returns:
            None
        """
        users_and_nodes = np.concatenate((self.users, self.nodes))
        for node in self.nodes:
            for comp in users_and_nodes:
                if node != comp:
                    self._addEdgeIfReaches(node, comp)

    def _addUserEdges(self) -> None:
        """
        Adds edges between users and nodes.

        Returns:
            None
        """
        for user in self.users:
            for node in self.nodes:
                self._addEdgeIfReaches(user, node)

    def _addEdgeIfReaches(
        self, comp1: NetworkComponent, comp2: NetworkComponent
    ) -> None:
        """
        Adds an edge between two components if the capacity is greater than zero.

        Args:
            comp1 (NetworkComponent): First network component.
            comp2 (NetworkComponent): Second network component.

        Returns:
            None
        """
        if comp1.reaches(comp2):
            capacity = comp1.calcCapacity(comp2)
            self.graph.add_edge(comp1, comp2, capacity=capacity)

    def updateCapacityMatrix(self) -> None:
        """Updates the capacity matrix to the most recent."""
        capacity_matrix: csr_array = nx.adjacency_matrix(
            G=self.graph,
            nodelist=np.concatenate((self.nodes, self.users)),
            dtype=int,
            weight="capacity",
        )
        self.capacity_matrix = capacity_matrix.toarray()

    def routeOfferedLoads(self) -> None:
        """
        Routes the offered loads between users.

        Returns:
            None
        """
        # Shuffle to not have preference in case of sequential max flow
        shuffled_users = self.np_random.permutation(self.users)
        # Set sour sink demand tuple
        source_sink_demands = [
            (user, user.dest, user.offeredLoad) for user in shuffled_users if user.dest
        ]

        # Remove users from graph without loss of generality, because flow shoudn't be routed
        # Over a user only sent and received from the users.
        sources_sinks = set(ss[0] for ss in source_sink_demands) | set(
            ss[1] for ss in source_sink_demands
        )
        graph_copy = self.graph.copy()
        graph_copy.remove_nodes_from(
            [user for user in self.users if user not in sources_sinks]
        )

        # Solve to get the maximum flow value with the flow dicts
        routing_result = self.routing_func(self.graph.copy(), source_sink_demands)
        if len(routing_result) == 3:
            max_flow, flow_dict, per_commodity = routing_result
        else:
            # Backward-compat: routing funcs other than sequential_max_flow
            # (e.g. artifical_src_sink_max_flow, MCMF variants) still return
            # a 2-tuple and don't provide per-commodity attribution.
            max_flow, flow_dict = routing_result
            per_commodity = {}
        self.per_commodity_flow = per_commodity

        # Update traffic hosted
        for node in self.nodes:
            node: MANETNode
            traffic = np.sum(list(flow_dict[node].values()))
            node.hostTraffic(traffic)

        # Retain per-drone directional throughput (additive; does not affect
        # trafficHosted or the reward). egress must reconcile with the existing
        # trafficHosted write — seatbelt against wiring in/out backwards.
        self.throughput_per_drone = {}
        for node in self.nodes:
            node: MANETNode
            out_flow = float(np.sum(list(flow_dict[node].values())))
            in_flow = float(
                np.sum([flow_dict[u].get(node, 0.0) for u in flow_dict])
            )
            self.throughput_per_drone[node.id] = {"in": in_flow, "out": out_flow}
            # Seatbelt: our egress must equal the value hostTraffic just received.
            expected_out = float(np.sum(list(flow_dict[node].values())))
            assert abs(out_flow - expected_out) < 1e-6, (
                f"egress mismatch for {node.id}: {out_flow} vs {expected_out}"
            )

        # Update data delivered
        for user, _, _ in source_sink_demands:
            user: User
            data_sent = np.sum(list(flow_dict[user].values()))
            user.setDataDelivered(data_sent)

        # Update the throughput
        self.networkThroughput = max_flow

    # * Getters
    def getThroughput(self) -> float:
        """
        Returns the network throughput.

        Returns:
            float: Network throughput.
        """
        return self.networkThroughput

    def getMaxThroughput(self) -> float:
        """
        Returns the maximum throughput of the network.

        Returns:
            float: Maximum throughput of the network.
        """
        return np.sum([user.offeredLoad for user in self.users])

    def getGraph(self) -> nx.DiGraph:
        """
        Returns the graph.

        Returns:
            nx.DiGraph: Graph representing the network.
        """
        return self.graph

    def getCapacityMatrix(self) -> np.ndarray:
        """
        Returns the capacity matrix.

        Returns:
            np.ndarray: Capacity matrix.
        """
        return self.capacity_matrix

    # * Setters
    def set_routing_func(self, routing_func: Callable) -> None:
        """Set the routing function.

        Args:
            routing_func (Callable): Function that routes the offered loads.
        """

        self.routing_func = routing_func
