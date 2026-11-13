// Numeric adapter for deck_context.project. No simulator transitions or cache.
#pragma once
#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>

namespace sts::deck_observation {
namespace py = pybind11;
class DeckContextEncoder {
    std::vector<int> cards_;
    int vocab_, slots_;
    double upgrade_scale_, misc_scale_;
public:
    DeckContextEncoder(std::vector<int> cards, int vocab, int slots,
                       double upgrade_scale, double misc_scale)
        : cards_(std::move(cards)), vocab_(vocab), slots_(slots),
          upgrade_scale_(upgrade_scale), misc_scale_(misc_scale) {
        if (vocab <= 0 || slots < Deck::MAX_SIZE || !(upgrade_scale > 0) || !(misc_scale > 0)
            || !std::isfinite(upgrade_scale) || !std::isfinite(misc_scale))
            throw std::invalid_argument("invalid deck projection geometry");
        for (int card: cards_) if (card < -1 || card >= vocab_)
            throw std::invalid_argument("deck projection vocabulary index out of range");
    }
    py::dict encode(const std::vector<const GameContext*>& games,
                    const std::vector<const BattleContext*>& battles) const {
        if (games.size() != battles.size()) throw std::invalid_argument("deck projection batch mismatch");
        const py::ssize_t n = games.size();
        py::array_t<float> counts({n, py::ssize_t(3), py::ssize_t(vocab_)});
        py::array_t<float> upgrades({n, py::ssize_t(3), py::ssize_t(vocab_)});
        py::array_t<float> misc({n, py::ssize_t(slots_)});
        float *cp = counts.mutable_data(), *up = upgrades.mutable_data(), *mp = misc.mutable_data();
        std::fill_n(cp, counts.size(), 0.f); std::fill_n(up, upgrades.size(), 0.f);
        std::fill_n(mp, misc.size(), 0.f);
        {
            py::gil_scoped_release release;
            for (py::ssize_t row = 0; row < n; ++row) {
                if (!games[row] || !battles[row]) throw std::invalid_argument("missing live deck projection root");
                const auto &gc = *games[row]; const auto &bc = *battles[row];
                const int deck_size = gc.deck.size();
                if (deck_size > slots_) throw std::invalid_argument("master deck exceeds observation capacity");
                float *m = mp + row * slots_;
                for (int i = 0; i < deck_size; ++i) m[i] = float(gc.deck.cards[i].misc / misc_scale_);
                // Same physical master-slot overlay and traversal order as
                // permanent_card_misc; temporary upgrades are never persisted.
                const auto overlay = [&](const CardInstance &card) {
                    if (card.getId() == CardId::RITUAL_DAGGER && card.uniqueId >= 0
                        && card.uniqueId < deck_size)
                        m[card.uniqueId] = float(card.specialData / misc_scale_);
                };
                int zone = 0;
                for (const auto *pile : {&bc.cards.drawPile, &bc.cards.discardPile, &bc.cards.exhaustPile}) {
                    float *c = cp + (row * 3 + zone) * vocab_;
                    float *u = up + (row * 3 + zone) * vocab_;
                    for (const auto &card: *pile) {
                        const int id = int(card.getId());
                        if (id < 0 || id >= int(cards_.size())) throw std::invalid_argument("unmapped native card");
                        const int index = cards_[id], count = card.getUpgradeCount();
                        if (index < 0) throw std::invalid_argument("invalid card in a public pile");
                        c[index] += count > 0;
                        // Python reference adds a double then stores float32
                        // after each card, not once after aggregating a pile.
                        u[index] = float(double(u[index]) + double(count) / upgrade_scale_);
                        overlay(card);
                    }
                    ++zone;
                }
                for (int i = 0; i < bc.cards.cardsInHand; ++i) overlay(bc.cards.hand[i]);
            }
        }
        py::dict result; result["pile_upgraded_counts"] = counts;
        result["pile_upgrade_sums"] = upgrades; result["outer_deck_misc"] = misc;
        return result;
    }
};
inline void bindDeckContextEncoder(py::module_ &m) {
    py::class_<DeckContextEncoder>(m, "DeckContextEncoder")
        .def(py::init<std::vector<int>, int, int, double, double>())
        .def("encode", &DeckContextEncoder::encode);
}
}
