# Numpy
import numpy as np

# Testing libs
import pytest
from unittest.mock import patch

# Own classes
from MANET.Network import Network
import MANET.Components as net_comps

import OptimumFinder.ExhaustiveSearch as es
from OptimumFinder.MINLP import findBestPositions

# For MINLP
from pyomo.environ import *

# For exhaustive search
from shapely.geometry import Polygon, Point


def test_findOptimumPosition():
    # Minimal input parameters
    nodes = np.array(
        [
            net_comps.MANETNode(id="n0", pos=np.array([22, 22])),
            net_comps.MANETNode(id="n1", pos=np.array([28, 28])),
        ]
    )
    users = np.array(
        [
            net_comps.User(id="u0", pos=np.array([20, 20])),
            net_comps.User(id="u1", pos=np.array([30, 30])),
        ]
    )
    user1, user2 = users
    user1: net_comps.User
    user2: net_comps.User
    user1.offeredLoad = 30e6
    user2.offeredLoad = 30e6
    user1.dest = user2
    user2.dest = user1

    jammers = np.array([net_comps.Jammer(id="j0", pos=np.array([40, 40]))])

    obj_val = None
    optimized_positions = None

    try:
        obj_val, optimized_positions = findBestPositions(
            nodes=nodes,
            users=users,
            jammers=jammers,
        )
        assert isinstance(obj_val, float)
        assert isinstance(optimized_positions, np.ndarray)
        assert optimized_positions.shape == (len(nodes), 2)
    except Exception as e:
        print(f"Objective value: {obj_val}")
        print(f"Optimized positions: {optimized_positions}")
        pytest.fail(f"findOptimumPosition raised an exception: {e}")


def test_ipoptWorks():
    # Create a simple Pyomo model
    model = ConcreteModel()
    model.x = Var(bounds=(0, 10), initialize=5)  # Variable with bounds
    model.y = Var(bounds=(0, 10), initialize=5)  # Another variable

    # Objective: Minimize x^2 + y^2
    model.obj = Objective(expr=model.x**2 + model.y**2, sense=minimize)

    # Constraint: x + y >= 8
    model.con = Constraint(expr=model.x + model.y >= 8)

    # Solve with IPOPT
    solver = SolverFactory("ipopt")
    result = solver.solve(model, tee=True)

    # Ensure solver terminated successfully
    assert (result.solver.status == SolverStatus.ok) or (
        result.solver.status == SolverStatus.warning
    ), "Solver did not terminate successfully!"

    # Ensure solver found an optimal solution
    assert (
        result.solver.termination_condition == TerminationCondition.optimal
    ), "Solver did not find an optimal solution! Termination condition: " + str(
        result.solver.termination_condition
    )

    # Validate the solution
    x_val = model.x.value
    y_val = model.y.value
    assert abs(x_val - 4) <= 1e-4, f"x value incorrect: {x_val}"
    assert abs(y_val - 4) <= 1e-4, f"y value incorrect: {y_val}"

    # Validate the objective value
    expected_obj = 32  # 4^2 + 4^2
    actual_obj = model.obj()
    assert (
        abs(actual_obj - expected_obj) <= 1e-4
    ), f"Objective value incorrect: {actual_obj}"


@pytest.fixture
def getExhaustiveSearcher() -> es.ExhaustiveSearcher:
    users = np.array(
        [
            net_comps.User(id="u0", pos=np.array([20, 20])),
            net_comps.User(id="u1", pos=np.array([30, 30])),
        ]
    )
    user1, user2 = users
    user1: net_comps.User
    user2: net_comps.User
    user1.offeredLoad = 30e6
    user2.offeredLoad = 30e6
    user1.dest = user2
    user2.dest = user1

    jammers = np.array([net_comps.Jammer(id="j0", pos=np.array([40, 40]))])

    nodes = np.array(
        [
            net_comps.MANETNode(id=f"n1", pos=np.array([50, 50])),
            net_comps.MANETNode(id=f"n2", pos=np.array([50, 50])),
        ]
    )
    nodes = np.array(nodes)

    exhaustiveSearcher = es.ExhaustiveSearcher(
        nodes, jammers, users, gridResolution=20, refinementResolution=10, maxLength=3
    )

    return exhaustiveSearcher


class Test_ExhaustiveSearch:
    def test_getConvexHull(self):
        points = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        hull = es.getConvexHull(points)
        expected_hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert hull.equals(expected_hull), f"Expected {expected_hull}, got {hull}"

    def test_joinPolygonWithCircle(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        circle = ((1, 1), 1)
        result = es.joinPolygonWithCircle(polygon, circle)
        expected_circle = Point(1, 1).buffer(1)
        expected_result = polygon.union(expected_circle)
        assert result.equals(
            expected_result
        ), f"Expected {expected_result}, got {result}"

    def test_subtractCircleFromPolygon(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        circle = ((1, 1), 1)
        result = es.subtractCircleFromPolygon(polygon, circle)
        expected_circle = Point(1, 1).buffer(1)
        expected_result = polygon.difference(expected_circle)
        assert result.equals(
            expected_result
        ), f"Expected {expected_result}, got {result}"

    def test_generateGrid(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        resolution = 5
        grid_points = es.generateGrid(polygon, resolution)
        expected_points = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.0],
                [2.0, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [1.5, 0.5],
                [2.0, 0.5],
                [0.0, 1.0],
                [0.5, 1.0],
                [1.0, 1.0],
                [1.5, 1.0],
                [2.0, 1.0],
                [0.0, 1.5],
                [0.5, 1.5],
                [1.0, 1.5],
                [1.5, 1.5],
                [2.0, 1.5],
                [0.0, 2.0],
                [0.5, 2.0],
                [1.0, 2.0],
                [1.5, 2.0],
                [2.0, 2.0],
            ]
        )
        assert np.array_equal(
            np.sort(grid_points, axis=0), np.sort(expected_points, axis=0)
        ), f"Expected {expected_points}, got {grid_points}"

    @pytest.mark.skip(reason="Change to allocation generator, test needs to be updated")
    def test_allocateNodes(self):
        grid_points = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]  # Example grid points
        num_nodes = 3  # Number of nodes to allocate

        # Differentiable case
        allocations_distinguishable = es.allocateNodes(
            grid_points, num_nodes, distinguishable=True
        )
        assert len(allocations_distinguishable) == 60  # 5P3 = 5! / (5 - 3)!
        assert all(
            len(allocation) == num_nodes for allocation in allocations_distinguishable
        )

        # Non-distinguishable case
        allocations_non_distinguishable = es.allocateNodes(
            grid_points, num_nodes, distinguishable=False
        )
        assert len(allocations_non_distinguishable) == 10  # 5C3 = 5! / (3!(5 - 3)!)
        assert all(
            len(allocation) == num_nodes
            for allocation in allocations_non_distinguishable
        )

        # Specific example checks (optional)
        assert (1, 1) in allocations_distinguishable[0]
        assert (1, 1) in allocations_non_distinguishable[0]

        # Edge case: num_nodes greater than available grid points
        with pytest.raises(ValueError):
            es.allocateNodes(grid_points, len(grid_points) + 1, distinguishable=True)

    def test_findTopAllocations(self, getExhaustiveSearcher: es.ExhaustiveSearcher):
        # Create a network using the fixture
        exhaustiveSearcher: es.ExhaustiveSearcher = getExhaustiveSearcher
        exhaustiveSearcher.max_possible_throughput = np.inf

        # Define mock allocations (positions for the nodes)
        grid_points = [
            (10, 10),
            (20, 20),
            (30, 30),
            (40, 40),
            (50, 50),
            (60, 60),
        ]
        num_nodes = len(exhaustiveSearcher.nodes)

        # Generate all possible allocations (for simplicity, assume non-distinguishable nodes)
        allocations = es.allocateNodes(
            grid_points=grid_points, num_nodes=num_nodes, distinguishable=False
        )

        # Mock the sampleThroughput function
        with patch(
            "OptimumFinder.ExhaustiveSearch.ExhaustiveSearcher.sampleThroughput"
        ) as mock_sampleThroughput:
            # Define the mock behavior: throughput based on the sum of node positions
            def mock_throughput():
                return sum(
                    node.pos[0] + node.pos[1] for node in exhaustiveSearcher.nodes
                )

            mock_sampleThroughput.side_effect = mock_throughput

            # Run findTopAllocations with a maximum length of 3
            maxLength = 3
            top_allocations = exhaustiveSearcher.findTopAllocations(allocations)
            top_allocations = [allocation for throughput, allocation in top_allocations]

            # Assertions
            assert (
                len(top_allocations) == maxLength
            ), "Top allocations count does not match maxLength."
            assert all(
                isinstance(allocation, tuple) and len(allocation) == num_nodes
                for allocation in top_allocations
            ), "Allocations format or length is incorrect."

            # Verify that the allocations are sorted in descending order of throughput
            throughputs = [
                sum(pos[0] + pos[1] for pos in allocation)
                for allocation in top_allocations
            ]
            assert throughputs == sorted(
                throughputs, reverse=True
            ), "Allocations are not sorted by throughput."
