# Components.py
In [Components.py](#MANET.Components) the core network components for the simulation are modeled. The main functionalities are modelled in [NetworkComponent](#MANET.Components.NetworkComponent) from which all, `MANETNode`, `Jammer` and `User` inherit and add their individual functionalities.

__NetworkComponent__

Includes methods for calculating signal transmission, capacity, and component interactions. The signal power transmitted is calculated using the simplified `Friis Equation`, where equal Gain and additional path loss are assumed. For the transmission capacity, a link with capacity being the minimum from the capacity deduced by `Shannon's Equation` and the `Nyquist Theorem` is considered.

Following parameters are used:

- `BANDWIDTH`: 20 MHz (2.4 GHz band channels typically have Bandwidth of 20 MHz or 40 MHz)
- `FREQUENCY`: 2.4 GHz (2.4 GHz band channel)
- `SPEED_OF_LIGHT`: 300,000,000 m/s (Speed of light in m/s)
- `PATH_LOSS_EXPONENT`: 1.8 (Adjust based on environment) -> Here the additional path loss. So, if used in the equation of the paper it would be $\lambda =$ `PATH_LOSS_EXPONENT`$+ 2$
- `ENVIRONMENT_NOISE`: 1e-6 mW (Environmental noise level)

__Components__

The resulting components used in the simulation are:

- [MANETNode](#MANET.Components.MANETNode): Which has the exact same functionality as the `NetworkComponent`.
- [Jammer](#MANET.Components.Jammer): Has additional functions to instead of transmit signal, jam components/transmit noise, which the `NetworkComponent` takes as `interference`.
- [User](#MANET.Components.User): A User additionaly, has a `destination` and an `offered load`. So, at each time step in the simulation it wants to send an `offered load` to its `destination`, which is another User.

::: MANET.Components
    options:
      show_root_heading: false  # Avoid showing the module name as a header
