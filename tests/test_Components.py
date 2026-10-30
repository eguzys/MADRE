# Need for testing
import pytest
from unittest.mock import patch, MagicMock

# Standard libraries
import numpy as np

# Own project libraries
import MANET.Components as net_comps


class Test_NetworkComponent:

    def test_approxRadius_method(self):
        """Check that the radius is as expected and that the link capacity at the radius' distance is
        exactly the capcity threshold used in the equation.
        """
        comp1 = net_comps.NetworkComponent(
            id="comp 1", signalPower_mW=500, pos=np.array([0, 0])
        )

        # Check radius
        radius = comp1.radius

        assert radius > 30
        assert radius < 100

        # Set second component at distance d=comp1.radius from comp1
        comp2 = net_comps.NetworkComponent(
            id="comp 2", signalPower_mW=200, pos=np.array([0, radius])
        )
        linkCap_1_to_2 = comp1.calcCapacity(comp2)

        assert np.isclose(
            linkCap_1_to_2, 500e3, atol=1e-2
        ), "Should equal threshold capacity."

    def test_reaches_method(self):
        """
        Check that a component only reaches another if it is in its radius.
        """
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=1
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 2", pos=np.array([5, 0]), signalPower_mW=1
        )
        comp3 = net_comps.NetworkComponent(
            id="comp 3", pos=np.array([25, 25]), signalPower_mW=1000
        )
        # comp1 and cop2 should have a small radius
        assert comp1.radius < 25
        assert comp2.radius < 25

        assert comp1.reaches(comp2)
        assert comp2.reaches(comp1)

        # So, comp1 has to small of a radius to reach comp3, but comp3 has larger
        # signal power and hence can reach comp2
        assert comp3.reaches(comp1), "Comp3 should reach comp1"
        assert not comp1.reaches(comp3), "Comp1 shouldn't reach comp3."

    def test_calcSignalTransmitted_if_in_reach(self):
        """
        Test that the signal transmitted between two components is as expected
        when they are within reach.
        """
        signalPow_set = 1

        # Create components with the same/close position and signal power
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=signalPow_set
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 2", pos=np.array([0, 0]), signalPower_mW=signalPow_set
        )
        comp3 = net_comps.NetworkComponent(
            id="comp 3", pos=np.array([0, 5]), signalPower_mW=signalPow_set
        )

        # Calculate signal transmitted from comp1 to comp2
        signal_comp1_to_comp2 = comp1.calcSignalTransmitted(comp2)

        assert (
            signal_comp1_to_comp2 == signalPow_set
        ), "Signal shouldn't decrease since comp1.pos = comp2.pos"

        # Calculate signal transmitted from comp1 to comp3
        signal_comp1_to_comp3 = comp1.calcSignalTransmitted(comp3)
        assert (
            signal_comp1_to_comp3 > 0 and signal_comp1_to_comp3 < signalPow_set
        ), "comp1.pos != comp3.pos, but still in reach. Hence, signal received should be some positive number"

    def test_calcSignalTransmitted_if_not_in_reach(self):
        """
        Test that the signal transmitted between two components is zero
        when they are not within reach.
        """
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=1
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 2", pos=np.array([40, 0]), signalPower_mW=1
        )

        # Check if components are within reach
        reaches_1_2 = comp1.reaches(comp2)
        reaches_2_1 = comp2.reaches(comp1)

        assert not reaches_1_2, "Component 1 should not reach Component 2"
        assert not reaches_2_1, "Component 2 should not reach Component 1"

        # Calculate signal transmitted from comp1 to comp2
        signal_comp1_to_comp2 = comp1.calcSignalTransmitted(comp2)

        # Since comp1 doesn't reach comp2, comp2 shouldn't receive any signal from comp1
        assert (
            signal_comp1_to_comp2 == 0
        ), "comp2 shouldn't receive any signal from comp1"

    def test_transmitSignal(self):
        """
        Test that a component can transmit its signal to another component.
        """
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=10
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 2", pos=np.array([0, 0]), signalPower_mW=5
        )

        # Transmit signal from comp1 to comp2
        comp1.transmitSignal(comp2)

        # Should receive exactly comp1.signalPower_mW as distance = 0
        assert (
            comp2.signalReceived_mW == 10
        ), "Should receive exactly comp1.signalPower_mW as distance = 0."

    def test_hostTraffic(self):
        """
        Test that a component can host traffic and reset its traffic count.
        """
        # Initialize
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=10
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 2", pos=np.array([0, 0]), signalPower_mW=5
        )

        # Host traffic on comp1
        comp1.hostTraffic(100)
        comp1.hostTraffic(10)

        assert (
            comp1.trafficHosted == 110
        ), "Traffic hosted should sum to 110 (= 10 + 100)"

        # Host traffic on comp2
        comp2.hostTraffic(10)

        assert (
            comp2.trafficHosted == 10
        ), "Traffic hosted should be 10, since only 10 was hosted."

        # Reset all values on comp1
        comp1.resetAll()

        assert comp1.trafficHosted == 0, "Traffic hosted should be reset to 0"
        assert comp2.trafficHosted == 10, "Traffic hosted should be reset to 0"

        # Reset traffic hosted on comp2
        comp2.resetTrafficHosted()

        assert (
            comp2.trafficHosted == 0
        ), f"Expected traffic hosted to be 0 after reset, but got {comp2.trafficHosted}"

    def test_calcCapacity(self):
        """
        Test that the capacity equal the Bandwith if the signal received equals the environment noise
        and if there is no interference.
        """
        comp1 = net_comps.NetworkComponent(
            id="comp 1",
            pos=np.array([0, 0]),
            signalPower_mW=net_comps.ENVIRONMENT_NOISE,
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 2",
            pos=np.array([0, 0]),
            signalPower_mW=net_comps.ENVIRONMENT_NOISE,
        )

        # Check that the capacity on the edge (comp1, comp2) is equal to the bandwidth
        capacity = comp1.calcCapacity(comp2)

        assert capacity == comp1.BANDWIDTH, "Capacity should equal bandwith as SINR = 1"

    def test_resetInterferenceReceived(self):
        """
        Test that the interference received by a component can be reset.
        """
        comp = net_comps.NetworkComponent(id="comp 1", pos=np.array([0, 0]))
        comp.interferenceReceived_mW = 100

        assert comp.interferenceReceived_mW == 100, "Interference was set to 100"

        comp.resetInterferenceReceived()
        assert comp.interferenceReceived_mW == 0, "Interference should be reset to 0"

    def test_resetSignalReceived(self):
        """
        Test that the signal received by a component can be reset.
        """
        comp = net_comps.NetworkComponent(id="comp 1", pos=np.array([0, 0]))
        comp.signalReceived_mW = 100

        assert comp.signalReceived_mW == 100, "Signal received was set to 100"

        comp.resetSignalReceived()
        assert comp.signalReceived_mW == 0, "Signal received was reset to 0"

    def test_resetTrafficHosted(self):
        """
        Test that the traffic hosted by a component can be reset.
        """
        # TODO: Implement test logic
        pass

    def test_resetAll(self):
        """
        Test that all reset methods are called (not worth testing in detail).
        """
        pass

    def test_repr(self):
        """
        Test the string representation of a component (not worth testing in detail).
        """
        pass

    def test_eq(self):
        """
        Check that equal components are identified correctly. A component should have a unique id and
        cannot be at more than one position at once.
        """
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=1000
        )
        comp2 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=1000
        )
        comp3 = net_comps.NetworkComponent(
            id="comp 2", pos=np.array([1, 1]), signalPower_mW=500
        )
        comp4 = comp1
        assert comp1 == comp2, "comp1 should be equal to comp2, as same pos and id"
        assert comp1 != comp3, "comp1 not equal to comp3, as diff id"
        assert comp1 == comp4, "comp1 should equal comp2, as comp4 is assigned comp1"

    def test_hash(self):
        """
        The network component should be hashed over its id.
        """
        comp = net_comps.NetworkComponent(id="comp 1", pos=np.array([0, 0]))
        assert hash(comp) == hash("comp 1")


class Test_Jammer:

    def test_jamming(self):
        """Test that jammer correctly affects the interference another comp receives"""
        comp1 = net_comps.NetworkComponent(
            id="comp 1", pos=np.array([0, 0]), signalPower_mW=1000
        )
        jammer = net_comps.Jammer(
            id="Jammer 1", pos=np.array([0, 0]), signalPower_mW=1000
        )
        jammer.startJamming()

        assert comp1.interferenceReceived_mW == 0, "comp1 wasn't jammed yet"

        jammer.jamComps(comp1)

        assert (
            comp1.interferenceReceived_mW == 1000
        ), "comp1 should receive interference"
