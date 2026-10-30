//
// OM2GNet — the OM2G track's network.
//
// ╔════════════════════════════════════════════════════════════════════════╗
// ║  ⚠️  OM2G TRACK — NOT SmokeNet.  ⚠️                                     ║
// ║  SmokeNet  = OMNET track: 802.11g, 20 MHz, ~50 Mbps, omnetpp.ini.      ║
// ║  OM2GNet   = OM2G  track: IEEE 802.15.4, 2.8 MHz, 250 kbps, om2g.ini.  ║
// ║  Their results are NEVER comparable.  See code/om2g/OM2G_README.md.    ║
// ╚════════════════════════════════════════════════════════════════════════╝
//
// Structurally identical to SmokeNet — same DroneHost, same UserHost, same
// RLBridge, same jammer, same configurator — with exactly ONE difference:
// the radio medium is Ieee802154NarrowbandScalarRadioMedium instead of
// Ieee80211ScalarRadioMedium.  A separate network (rather than a `like`
// parameter on SmokeNet) so that the two tracks can never be selected by
// accident: choosing the network chooses the PHY, and the network name
// carries the track token.
//
// The medium's own defaults are overridden in om2g.ini: BreakpointPathLoss ->
// LogNormalShadowing(alpha=2.5, sigma=4), so propagation is byte-identical to
// the OMNET track and "only the channel capacity moved" is literally true.
//

package rlsmoke;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.wireless.ieee802154.packetlevel.Ieee802154NarrowbandScalarRadioMedium;

import rlsmoke.JammerHost;
import rlsmoke.RLBridge;
import rlsmoke.UserHost;


network OM2GNet
{
    parameters:
        int numDrones;
        int numUsers;
        @display("bgb=2200,2200");

    submodules:
        radioMedium: Ieee802154NarrowbandScalarRadioMedium {
            parameters:
                @display("p=-100,-100;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            parameters:
                config = default(xml("<config><interface hosts='*' address='10.0.x.x' netmask='255.255.0.0'/></config>"));
                @display("p=-100,-100;is=s");
        }
        rlBridge: RLBridge {
            parameters:
                numDrones = parent.numDrones;
                numUsers  = parent.numUsers;
                @display("p=-100,-100;is=s");
        }
        host[numDrones]: DroneHost {
            parameters:
                droneId = index;
                @display("i=device/pocketpc_s;r=,,#707070");
        }
        user[numUsers]: UserHost {
            parameters:
                userId = index;
                @display("i=device/cellphone_s;r=,,#CC3333");
        }
        jammer: JammerHost {
            parameters:
                @display("i=device/accesspoint_s;r=,,#FF0000");
        }

    connections allowunconnected:
}
