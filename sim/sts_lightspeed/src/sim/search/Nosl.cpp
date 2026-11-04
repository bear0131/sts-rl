#include "sim/search/Nosl.h"

#include <algorithm>

#include "game/Random.h"

namespace sts::search {

std::uint64_t noslMix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

namespace {
std::uint64_t streamSeed(std::uint64_t nonce, std::uint64_t stream) {
    return noslMix(nonce ^ noslMix(stream));
}
}

void noslReseed(BattleContext &bc, std::uint64_t nonce) {
    bc.aiRng = sts::Random(streamSeed(nonce, 1));
    bc.cardRandomRng = sts::Random(streamSeed(nonce, 2));
    bc.miscRng = sts::Random(streamSeed(nonce, 3));
    bc.monsterHpRng = sts::Random(streamSeed(nonce, 4));
    bc.potionRng = sts::Random(streamSeed(nonce, 5));
    bc.shuffleRng = sts::Random(streamSeed(nonce, 6));
}

void redeterminize(BattleContext &bc, std::uint64_t nonce,
                   bool canonicalizeDrawPile) {
    noslReseed(bc, nonce);
    // The remaining-card multiset is observable but its order is hidden.
    // Canonicalization prevents the result from depending on the source
    // world's hidden order before sampling an independent permutation.
    if (canonicalizeDrawPile) {
        std::sort(
            bc.cards.drawPile.begin(),
            bc.cards.drawPile.end(),
            [](const CardInstance &lhs, const CardInstance &rhs) {
                return lhs.getUniqueId() < rhs.getUniqueId();
            }
        );
    }
    java::Collections::shuffle(
        bc.cards.drawPile.begin(),
        bc.cards.drawPile.end(),
        java::Random(streamSeed(nonce, 7))
    );
}

void noslRedeterminize(BattleContext &bc, std::uint64_t nonce) {
    redeterminize(bc, nonce, true);
}

void noslRedeterminizeCanonical(BattleContext &bc, std::uint64_t nonce) {
    redeterminize(bc, nonce, false);
}

}
