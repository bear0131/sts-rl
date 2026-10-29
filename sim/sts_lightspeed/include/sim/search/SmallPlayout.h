#pragma once
#include "sim/search/Action.h"
#include <array>
#include <memory>
#include <vector>

namespace sts::search {
constexpr int SMALL_STATE = 32, SMALL_ACTION = 24, SMALL_HIDDEN = 64, SMALL_EMBED = 32;
int smallKeyCount();
struct SmallActionFeatures {
    std::uint32_t bits;
    std::vector<int> keys;
    std::array<float, SMALL_ACTION> x{};
};
struct SmallFeatures {
    std::array<float, SMALL_STATE> state{};
    std::vector<SmallActionFeatures> actions;
};
SmallFeatures smallFeatures(const BattleContext &bc);
// Parameter order is the versioned Python export contract, not a Torch ABI.
class SmallPlayoutNet {
    std::vector<float> weights_;
public:
    explicit SmallPlayoutNet(std::vector<float> weights);
    std::vector<float> logits(const SmallFeatures &features) const;
};
struct SmallSearchConfig {
    int simulations = 250, playoutDepth = 200, treeDepth = 1024;
    double exploration = 0.8, temperature = 1.0;
};
struct SmallSearchStats {
    std::uint64_t simulations = 0, treeTransitions = 0, playoutTransitions = 0;
    std::uint64_t terminalWins = 0, terminalLosses = 0, terminalEscapes = 0;
    std::uint64_t depthCuts = 0, treeCuts = 0, chanceReplays = 0, nodes = 0;
    std::uint64_t policyCalls = 0, actionScores = 0;
    std::array<std::uint64_t, 201> playoutLengths{};
    double seconds = 0;
};
struct SmallSearchResult {
    std::uint32_t action;
    SmallFeatures features;
    std::vector<int> visits;
    std::vector<double> values;
    SmallSearchStats stats;
};
double smallManualValue(const BattleContext &bc);
SmallSearchResult smallSearch(const BattleContext &root, const SmallPlayoutNet &net,
                             const SmallSearchConfig &cfg, std::uint64_t nonce);
}
