# Need for testing
from unittest.mock import patch, MagicMock
import pytest

# Standard libraries
import numpy as np
import networkx as nx

# Own project libraries
import MANET.Components as comps
import MANET.Network as net
import MANET.Environments as envs
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw


@pytest.fixture
def createNetwork() -> net.Network:
    node_positions = np.array(
        [[0, 0], [25, 25], [50, 50], [75, 75], [100, 100]], dtype=np.float32
    )
    jammer_positions = np.array([[0, 0]], dtype=np.float32)
    user_positions = np.array(
        [[0, 0], [25, 25], [50, 50], [75, 75], [100, 100]], dtype=np.float32
    )

    network = createNetworkFrom(node_positions, jammer_positions, user_positions)

    return network


def createNetworkFrom(
    node_positions=[], jammer_positions=[], user_positions=[]
) -> net.Network:

    nodes = np.array(
        [
            comps.MANETNode(id=f"Node {i}", pos=pos)
            for i, pos in enumerate(node_positions)
        ]
    )
    jammers = np.array(
        [
            comps.Jammer(id=f"Jammer {i}", pos=pos)
            for i, pos in enumerate(jammer_positions)
        ]
    )
    users = np.array(
        [comps.User(id=f"User {i}", pos=pos) for i, pos in enumerate(user_positions)]
    )
    network = net.Network(nodes, jammers, users)

    return network


class Test_init:
    # very simple, no test needed
    pass


class Test_initializeGraph:
    # very simple, no test needed
    pass


class Test_updateNetwork:
    # better to test methods inside
    pass


class Test_resetThroughput:
    # to simple to test
    pass


class Test_resetValues:
    def test_allValuesAreReset(self, createNetwork):
        network: net.Network = createNetwork

        network.updateNetwork()

        network.resetValues()

        nodes, jammers, users = network.nodes, network.jammers, network.users

        for comp in nodes:
            node: comps.MANETNode = comp
            assert node.signalReceived_mW == 0
            assert node.interferenceReceived_mW == 0

        for comp in users:
            user: comps.User = comp
            assert user.signalReceived_mW == 0
            assert user.interferenceReceived_mW == 0

        for comp in jammers:
            jammer: comps.Jammer = comp
            assert jammer.signalReceived_mW == 0
            assert jammer.interferenceReceived_mW == 0


class Test_updateSignals:
    def test_checkSignalsCorrectlyUpdated(self):
        node_positions = np.array([[0, 0]], dtype=np.float32)
        jammer_positions = np.array([[0, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0], [0, 0]], dtype=np.float32)

        network = createNetworkFrom(node_positions, jammer_positions, user_positions)
        network.resetValues()
        network.updateSignals()

        node: comps.MANETNode = network.nodes[0]
        jammer: comps.Jammer = network.jammers[0]
        user1: comps.User = network.users[0]
        user2: comps.User = network.users[1]

        assert node.signalReceived_mW == user1.signalPower_mW + user2.signalPower_mW
        assert user1.signalReceived_mW == node.signalPower_mW
        assert user2.signalReceived_mW == node.signalPower_mW
        assert (
            jammer.signalReceived_mW
            == node.signalPower_mW + user1.signalPower_mW + user2.signalPower_mW
        )


class Test_updateJammersEffect:

    def test_seeIfJammingWorks(self):
        node_positions = np.array([[0, 0]], dtype=np.float32)
        jammer_positions = np.array([[0, 0], [0, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0]], dtype=np.float32)

        network = createNetworkFrom(node_positions, jammer_positions, user_positions)

        node: comps.MANETNode = network.nodes[0]
        jammer1: comps.Jammer = network.jammers[0]
        jammer2: comps.Jammer = network.jammers[1]
        user: comps.User = network.users[0]

        jammer1.startJamming()
        jammer2.startJamming()

        network.resetValues()

        all_comps = [node, jammer1, jammer2, user]
        for comp in all_comps:
            net_comp: comps.NetworkComponent = comp
            assert net_comp.interferenceReceived_mW == 0

        network.updateJammersEffect()

        assert (
            node.interferenceReceived_mW
            == jammer1.signalPower_mW + jammer2.signalPower_mW
        )
        assert (
            user.interferenceReceived_mW
            == jammer1.signalPower_mW + jammer2.signalPower_mW
        )
        assert jammer1.interferenceReceived_mW == 0
        assert jammer2.interferenceReceived_mW == 0


class Test_updateEdges:
    # not worth testing
    pass


class Test_addEdges:
    def test_simpleScenario(self):
        node_positions = np.array([[0, 0], [0, 0]], dtype=np.float32)
        jammer_positions = np.array([[0, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0], [0, 0]], dtype=np.float32)

        network = createNetworkFrom(node_positions, jammer_positions, user_positions)

        node1: comps.MANETNode = network.nodes[0]
        node2: comps.MANETNode = network.nodes[1]
        jammer: comps.Jammer = network.jammers[0]
        user1: comps.User = network.users[0]
        user2: comps.User = network.users[1]

        graph: nx.DiGraph = network.graph
        graph.clear_edges()

        network.addEdges()

        # user only connect to nodes
        assert not graph.has_edge(user1, user2)
        assert not graph.has_edge(user2, user1)

        assert graph.has_edge(user1, node1)
        assert graph.has_edge(user1, node2)
        assert graph.has_edge(user2, node1)
        assert graph.has_edge(user2, node2)

        # nodes connect to each other and the users
        assert graph.has_edge(node1, node2)
        assert graph.has_edge(node2, node1)

        assert graph.has_edge(node1, user1)
        assert graph.has_edge(node1, user2)
        assert graph.has_edge(node2, user1)
        assert graph.has_edge(node2, user2)

        # jammer is a jammer, so shouldn't make any connections
        all_comps = np.concatenate((network.nodes, network.jammers, network.users))

        for comp in all_comps:
            net_comp: comps.NetworkComponent = comp

            assert not graph.has_edge(jammer, net_comp)
            assert not graph.has_edge(net_comp, jammer)

    def test_capacitiesSetCorrectly(self):
        node_positions = np.array([[0, 0]], dtype=np.float32)
        jammer_positions = np.array([[0, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0]], dtype=np.float32)

        network = createNetworkFrom(node_positions, jammer_positions, user_positions)

        node: comps.MANETNode = network.nodes[0]
        jammer: comps.Jammer = network.jammers[0]
        user: comps.User = network.users[0]

        graph: nx.DiGraph = network.graph
        graph.clear_edges()

        network.addEdges()

        capacities = nx.get_edge_attributes(graph, "capacity")

        assert capacities[(node, user)] == node.calcCapacity(user)
        assert capacities[(user, node)] == user.calcCapacity(node)

    def test_nodeEdge(self):
        node_positions = np.array([[0, 0], [25, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0]], dtype=np.float32)

        network: net.Network = createNetworkFrom(
            node_positions=node_positions, user_positions=user_positions
        )

        node1: comps.MANETNode = network.nodes[0]
        node2: comps.MANETNode = network.nodes[1]
        user1: comps.User = network.users[0]

        graph: nx.DiGraph = network.graph
        graph.clear_edges()

        network.addEdges()

        # nodes connect to each other and the users
        assert graph.has_edge(node1, node2)
        assert graph.has_edge(node2, node1)

        capacities = nx.get_edge_attributes(graph, "capacity")

        assert capacities[(node1, node2)] == node1.calcCapacity(node2)
        assert capacities[(node1, node2)] >= 100e3


class Test_sendPackets:
    def test_trafficHosted_simpleScenario(self):
        """
        Test that the traffic hosted correctly is used.
        """
        # Initialize positions
        node_positions = np.array([[5, 5]], dtype=np.float32)
        user_positions = np.array([[0, 0], [10, 10]], dtype=np.float32)

        # Create network
        network: net.Network = createNetworkFrom(
            node_positions=node_positions, user_positions=user_positions
        )

        # Get actors
        user1, user2 = network.users
        user1: comps.User
        user2: comps.User
        node: comps.MANETNode = network.nodes[0]

        # Set offered load
        user1.setOfferedLoad(10e6)  # 10 Mbit
        user1.setDestination(user2)
        user2.setDestination(None)  # Ensure no destination is set

        network.updateNetwork()  # Ensure network is up to date

        # Check whether traffic hosted is correctly updated.
        network.routeOfferedLoads()

        assert user1.trafficHosted == 0
        assert user2.trafficHosted == 0

        assert node.trafficHosted == user1.offeredLoad

    def test_networkThroughputUpdated(self):
        # Initialize positions
        node_positions = np.array([[5, 5]], dtype=np.float32)
        jammer_positions = np.array([[0, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0], [10, 10]], dtype=np.float32)

        # Create network
        network = createNetworkFrom(node_positions, jammer_positions, user_positions)

        # Get actors
        node: comps.MANETNode = network.nodes[0]
        jammer: comps.Jammer = network.jammers[0]
        user1: comps.User = network.users[0]
        user2: comps.User = network.users[1]

        # Set offered loads
        user1.setOfferedLoad(23e6)
        user2.setOfferedLoad(13e6)

        # Set destinations
        user1.setDestination(user2)
        user2.setDestination(user1)

        # Check throughput is updated correctly
        network.updateNetwork()

        assert network.getThroughput() == 0

        network.routeOfferedLoads()

        assert network.getThroughput() == user1.offeredLoad + user2.offeredLoad
        assert network.getMaxThroughput() == user1.offeredLoad + user2.offeredLoad


class Test__addNodeEdges:
    # already covered
    pass


class Test__addUserEdges:
    # already covered
    pass


class Test__addEdgeIfCapacity:
    # already covered
    pass


class Test__getIndex:
    # very simple
    pass


class Test_getThroughput:
    # already covered
    pass


class Test_getMaxThroughput:
    # already covered
    pass


class Test_createGraph_method:
    # already covered
    pass


class Test_updateGraph_method:
    # already covered through its containing methods
    pass
