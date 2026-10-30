# Maths
import numpy as np

# Graph
import networkx as nx

# For testing
import pytest

# Own project libraries
import MANET.Components as comps
import MANET.Network as net
import MANET.Environments as envs
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw

# Tested
from MANET.Routing import MCMF, MaxFlow


class Test_Pulp_MCMF:

    def test_simple_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_1(
            G, source_sink_demands
        )
        assert max_value == 5
        assert flow_vars == {0: {1: 5}, 1: {2: 5}}

    def test_complex_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=5)
        G.add_edge(2, 3, capacity=5)
        G.add_edge(1, 3, capacity=5)
        source_sink_demand1 = (0, 3, 10)
        source_sink_demand2 = (3, 0, 10)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_1(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 10}, 1: {2: 5, 3: 5}, 2: {3: 5}}

    def test_no_flow(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=0)
        G.add_edge(1, 2, capacity=0)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_1(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(2, 3, capacity=10)
        source_sink_demand1 = (0, 3, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_1(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_circular_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        G.add_edge(2, 0, capacity=5)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demand2 = (2, 0, 5)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_1(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 5}, 1: {2: 5}, 2: {0: 5}}

    def test_scenario_with_multiple_loops(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("U1", "N1"),
                ("N1", "U1"),
                ("U2", "N2"),
                ("N2", "U2"),
                ("U3", "N3"),
                ("N3", "U3"),
                ("N1", "N2"),
                ("N2", "N1"),
                ("N2", "N3"),
                ("N3", "N2"),
            ]
        )

        users = ["U1", "U2", "U3"]
        nodes = ["N1", "N2", "N3"]
        # Set the capacity to be less than the offered load
        for user, node in zip(users, nodes):
            G[user][node]["capacity"] = 10
            G[node][user]["capacity"] = 10

        # Set the capacity to be less than the offered load
        G["N1"]["N2"]["capacity"] = 3
        G["N2"]["N1"]["capacity"] = 3
        G["N2"]["N3"]["capacity"] = 3
        G["N3"]["N2"]["capacity"] = 3

        source_sink_demand1 = ("U1", "U2", 5)
        source_sink_demand2 = ("U2", "U3", 5)
        source_sink_demand3 = ("U3", "U1", 5)
        source_sink_demands = [
            source_sink_demand1,
            source_sink_demand2,
            source_sink_demand3,
        ]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_1(
            G, source_sink_demands
        )
        assert max_value == 9
        assert flow_vars == {
            "U1": {
                "N1": 3,
            },
            "N1": {"N2": 3, "U1": 3},
            "N2": {"U2": 3, "N3": 3, "N1": 3},
            "U2": {"N2": 3},
            "N3": {"U3": 3, "N2": 3},
            "U3": {"N3": 3},
        }


class Test_Pulp_MCMF_v2:

    def test_simple_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_2(
            G, source_sink_demands
        )
        assert max_value == 5
        assert flow_vars == {0: {1: 5}, 1: {2: 5}}

    def test_complex_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=5)
        G.add_edge(2, 3, capacity=5)
        G.add_edge(1, 3, capacity=5)
        source_sink_demand1 = (0, 3, 10)
        source_sink_demand2 = (3, 0, 10)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_2(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 10}, 1: {2: 5, 3: 5}, 2: {3: 5}}

    def test_no_flow(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=0)
        G.add_edge(1, 2, capacity=0)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_2(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(2, 3, capacity=10)
        source_sink_demand1 = (0, 3, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_2(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_circular_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        G.add_edge(2, 0, capacity=5)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demand2 = (2, 0, 5)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_2(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 5}, 1: {2: 5}, 2: {0: 5}}

    def test_scenario_with_multiple_loops(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("U1", "N1"),
                ("N1", "U1"),
                ("U2", "N2"),
                ("N2", "U2"),
                ("U3", "N3"),
                ("N3", "U3"),
                ("N1", "N2"),
                ("N2", "N1"),
                ("N2", "N3"),
                ("N3", "N2"),
            ]
        )

        users = ["U1", "U2", "U3"]
        nodes = ["N1", "N2", "N3"]
        # Set the capacity to be less than the offered load
        for user, node in zip(users, nodes):
            G[user][node]["capacity"] = 10
            G[node][user]["capacity"] = 10

        # Set the capacity to be less than the offered load
        G["N1"]["N2"]["capacity"] = 3
        G["N2"]["N1"]["capacity"] = 3
        G["N2"]["N3"]["capacity"] = 3
        G["N3"]["N2"]["capacity"] = 3

        source_sink_demand1 = ("U1", "U2", 5)
        source_sink_demand2 = ("U2", "U3", 5)
        source_sink_demand3 = ("U3", "U1", 5)
        source_sink_demands = [
            source_sink_demand1,
            source_sink_demand2,
            source_sink_demand3,
        ]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_pulp_2(
            G, source_sink_demands
        )
        assert max_value == 9
        assert flow_vars == {
            "U1": {
                "N1": 3,
            },
            "N1": {"N2": 3, "U1": 3},
            "N2": {"U2": 3, "N3": 3, "N1": 3},
            "U2": {"N2": 3},
            "N3": {"U3": 3, "N2": 3},
            "U3": {"N3": 3},
        }


class Test_XPress_MCMF:

    def test_simple_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_xpress(
            G, source_sink_demands
        )
        assert max_value == 5
        assert flow_vars == {0: {1: 5}, 1: {2: 5}}

    def test_complex_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=5)
        G.add_edge(2, 3, capacity=5)
        G.add_edge(1, 3, capacity=5)
        source_sink_demand1 = (0, 3, 10)
        source_sink_demand2 = (3, 0, 10)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_xpress(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 10}, 1: {2: 5, 3: 5}, 2: {3: 5}}

    def test_no_flow(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=0)
        G.add_edge(1, 2, capacity=0)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_xpress(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(2, 3, capacity=10)
        source_sink_demand1 = (0, 3, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_xpress(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_circular_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        G.add_edge(2, 0, capacity=5)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demand2 = (2, 0, 5)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_xpress(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 5}, 1: {2: 5}, 2: {0: 5}}

    def test_scenario_with_multiple_loops(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("U1", "N1"),
                ("N1", "U1"),
                ("U2", "N2"),
                ("N2", "U2"),
                ("U3", "N3"),
                ("N3", "U3"),
                ("N1", "N2"),
                ("N2", "N1"),
                ("N2", "N3"),
                ("N3", "N2"),
            ]
        )

        users = ["U1", "U2", "U3"]
        nodes = ["N1", "N2", "N3"]
        # Set the capacity to be less than the offered load
        for user, node in zip(users, nodes):
            G[user][node]["capacity"] = 10
            G[node][user]["capacity"] = 10

        # Set the capacity to be less than the offered load
        G["N1"]["N2"]["capacity"] = 3
        G["N2"]["N1"]["capacity"] = 3
        G["N2"]["N3"]["capacity"] = 3
        G["N3"]["N2"]["capacity"] = 3

        source_sink_demand1 = ("U1", "U2", 5)
        source_sink_demand2 = ("U2", "U3", 5)
        source_sink_demand3 = ("U3", "U1", 5)
        source_sink_demands = [
            source_sink_demand1,
            source_sink_demand2,
            source_sink_demand3,
        ]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_xpress(
            G, source_sink_demands
        )
        assert max_value == 9
        assert flow_vars == {
            "U1": {
                "N1": 3,
            },
            "N1": {"N2": 3, "U1": 3},
            "N2": {"U2": 3, "N3": 3, "N1": 3},
            "U2": {"N2": 3},
            "N3": {"U3": 3, "N2": 3},
            "U3": {"N3": 3},
        }


class Test_Gurobi_MCMF:
    def test_simple_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_gurobi(
            G, source_sink_demands
        )
        assert max_value == 5
        assert flow_vars == {0: {1: 5}, 1: {2: 5}}

    def test_complex_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=5)
        G.add_edge(2, 3, capacity=5)
        G.add_edge(1, 3, capacity=5)
        source_sink_demand1 = (0, 3, 10)
        source_sink_demand2 = (3, 0, 10)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_gurobi(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 10}, 1: {2: 5, 3: 5}, 2: {3: 5}}

    def test_no_flow(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=0)
        G.add_edge(1, 2, capacity=0)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_gurobi(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(2, 3, capacity=10)
        source_sink_demand1 = (0, 3, 5)
        source_sink_demands = [source_sink_demand1]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_gurobi(
            G, source_sink_demands
        )
        assert max_value == 0
        assert flow_vars == {}

    def test_circular_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=10)
        G.add_edge(2, 0, capacity=5)
        source_sink_demand1 = (0, 2, 5)
        source_sink_demand2 = (2, 0, 5)
        source_sink_demands = [source_sink_demand1, source_sink_demand2]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_gurobi(
            G, source_sink_demands
        )
        assert max_value == 10
        assert flow_vars == {0: {1: 5}, 1: {2: 5}, 2: {0: 5}}

    def test_scenario_with_multiple_loops(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("U1", "N1"),
                ("N1", "U1"),
                ("U2", "N2"),
                ("N2", "U2"),
                ("U3", "N3"),
                ("N3", "U3"),
                ("N1", "N2"),
                ("N2", "N1"),
                ("N2", "N3"),
                ("N3", "N2"),
            ]
        )

        users = ["U1", "U2", "U3"]
        nodes = ["N1", "N2", "N3"]
        # Set the capacity to be less than the offered load
        for user, node in zip(users, nodes):
            G[user][node]["capacity"] = 10
            G[node][user]["capacity"] = 10

        # Set the capacity to be less than the offered load
        G["N1"]["N2"]["capacity"] = 3
        G["N2"]["N1"]["capacity"] = 3
        G["N2"]["N3"]["capacity"] = 3
        G["N3"]["N2"]["capacity"] = 3

        source_sink_demand1 = ("U1", "U2", 5)
        source_sink_demand2 = ("U2", "U3", 5)
        source_sink_demand3 = ("U3", "U1", 5)
        source_sink_demands = [
            source_sink_demand1,
            source_sink_demand2,
            source_sink_demand3,
        ]
        max_value, flow_vars = MCMF.multi_commodity_max_flow_gurobi(
            G, source_sink_demands
        )
        assert np.isclose(max_value, 9, rtol=1e-4)
        assert flow_vars == {
            "U1": {
                "N1": 3,
            },
            "N1": {"N2": 3, "U1": 3},
            "N2": {"U2": 3, "N3": 3, "N1": 3},
            "U2": {"N2": 3},
            "N3": {"U3": 3, "N2": 3},
            "U3": {"N3": 3},
        }


class Test_MaxFlow_NX:
    def test_simple_scenario(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, capacity=10)
        G.add_edge(1, 2, capacity=5)
        G.add_edge(2, 3, capacity=5)
        G.add_edge(1, 3, capacity=5)
        max_value, flow_vars = nx.maximum_flow(G, 0, 3)
        assert max_value == 10
        assert flow_vars == {0: {1: 10}, 1: {2: 5, 3: 5}, 2: {3: 5}, 3: {}}


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


class Test_sequentialMaxFlow:

    def test_packetRoutedCorrectly(self):
        # Initialize the positions
        node_positions = np.array([[5, 5]], dtype=np.float32)
        jammer_positions = np.array([[0, 0]], dtype=np.float32)
        user_positions = np.array([[0, 0], [10, 10]], dtype=np.float32)

        # Create network that makes the connections
        network = createNetworkFrom(node_positions, jammer_positions, user_positions)

        # Get actors
        node: comps.MANETNode = network.nodes[0]
        jammer: comps.Jammer = network.jammers[0]
        user1: comps.User = network.users[0]
        user2: comps.User = network.users[1]

        # Set offered load
        user1.setOfferedLoad(100e6)

        network.updateNetwork()  # Ensure network is up to date

        # Check wheter capacities are correctly reduced
        graph: nx.DiGraph = network.graph

        capacities_prev = nx.get_edge_attributes(graph, "capacity")

        source_sink_demands = [(user1, user2, user1.offeredLoad)]
        max_flow_value, flow_dict = MaxFlow.sequential_max_flow(
            graph, source_sink_demands
        )

        capacities_now = nx.get_edge_attributes(graph, "capacity")
        assert (
            capacities_now[(user1, node)]
            == capacities_prev[(user1, node)] - user1.offeredLoad
        )
        assert (
            capacities_now[(node, user2)]
            == capacities_prev[(node, user2)] - user1.offeredLoad
        )


def compare_flow_dicts(actual, expected, rtol=1e-5, atol=1e-8):
    """Compare two flow dictionaries within numerical tolerance."""
    if actual.keys() != expected.keys():
        return False

    for source in expected:
        if actual[source].keys() != expected[source].keys():
            return False
        for target in expected[source]:
            if not np.isclose(
                actual[source][target], expected[source][target], rtol=rtol, atol=atol
            ):
                return False
    return True
