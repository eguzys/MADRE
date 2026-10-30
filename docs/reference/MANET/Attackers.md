# Attackers.py
To model "realistic" attackers, different attacker models have been created in [Attackers.py](#MANET.Attackers). The attacker models control the jammers and bring them into a good position to start jamming. A jammer only starts jamming, once it was activated.

__Attackers__

There are three levels of attackers which differ in there knowledge about the network formed by the [Users](Components.md#MANET.Components.User) and [MANETNodes](Components.md/#MANET.Components.MANETNode). So, it is differentiated between no knowledge, only knowledge about the topology and knowledge about the whole network and for each their is an AttackerModel with a strategy corresponding to its knowledge. 

Further, for all three `AttackerModels` they can either be `static` or `moving`, meaning that as soon as they start jamming, if they are `static`, they stop moving or if there are `moving` jammers the keep moving.

- [GreedyJammers](#MANET.Attackers.GreedyJammers): The `GreedyJammers` move towards the direction with the highest signal strength received. The jammers sample signals from multiple directions (N, NE, E, SE, ...) and choose the direction with the highest signal. They continue moving in that direction while periodically sampling adjacent directions to stay on the right path. `GreedyJammers` do not have knowledge of the network topology and find the best spot by sampling signals.
- [ClusterJammers](#MANET.Attackers.ClusterJammers): The `ClusterJammers` know of the network topology and want to jam the biggest user cluster. Hence, they move to the largest user clusters to jam them. The first jammer moves to the center of the largest user cluster, the second to the second largest, etc.
- [TrafficJammers](#MANET.Attackers.TrafficJammers): The `TrafficJammers` are supposed to know all about the networks topology. Hence, the try to jam the nodes that route the most data

::: MANET.Attackers
    options:
      show_root_heading: false  # Avoid showing the module name as a header
