// JammerNoiseSource.h
//
// Extends INET's NoiseSource with deferred activation.
// The noise source is registered with the radio medium at init but emits
// nothing until activateJammer() is called. Once called, it transmits
// continuously (NoiseTransmitter duration >> episode length, sleepInterval ~ 0).
//
// Choice rationale: INET's NoiseSource already bypasses CSMA/CA entirely —
// it registers directly as an IRadio and pushes noise transmissions to the
// medium without any Mac layer. This is the simplest correct approach.
//
#ifndef JAMMERNOISESOURCE_H
#define JAMMERNOISESOURCE_H

#include "inet/physicallayer/wireless/noise/NoiseSource.h"

namespace inet {

class INET_API JammerNoiseSource : public physicallayer::NoiseSource
{
  protected:
    bool activated_ = false;

    // Override to suppress automatic sleep→transmit cycle until activated.
    virtual void scheduleSleepTimer() override;

  public:
    // Called by JammerGradientMobility when step >= activate.
    void activateJammer();
};

} // namespace inet

#endif
