#ifndef STS_LIGHTSPEED_NOSL_H
#define STS_LIGHTSPEED_NOSL_H

#include <cstdint>

#include "combat/BattleContext.h"

namespace sts::search {

// Mixes an opaque search nonce without exposing or consuming the live
// simulator RNG streams. This is the stream root for NOSL samples.
std::uint64_t noslMix(std::uint64_t value);

// Re-determinize only hidden future randomness for one search simulation.
// The observable multiset of cards remaining in the draw pile is preserved,
// while its hidden order is replaced by an independent Java-compatible
// permutation. The caller must pass a copy of the live BattleContext.
void noslRedeterminize(BattleContext &bc, std::uint64_t nonce);

}

#endif // STS_LIGHTSPEED_NOSL_H
