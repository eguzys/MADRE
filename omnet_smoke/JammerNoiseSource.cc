// JammerNoiseSource.cc
#include "JammerNoiseSource.h"

namespace inet {

Define_Module(JammerNoiseSource);

void JammerNoiseSource::scheduleSleepTimer()
{
    // Before activation: suppress the sleep→transmit cycle.
    // After activation: delegate to parent so transmission restarts after each burst.
    if (!activated_) return;
    physicallayer::NoiseSource::scheduleSleepTimer();
}

void JammerNoiseSource::activateJammer()
{
    if (activated_) return;
    activated_ = true;
    startTransmission();
}

} // namespace inet
