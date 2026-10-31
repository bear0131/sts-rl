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
    std::array<float, SMALL_STATE * SMALL_HIDDEN> stateColumns_;
    std::array<float, SMALL_HIDDEN * SMALL_EMBED> hiddenColumns_;
public:
    explicit SmallPlayoutNet(std::vector<float> weights);
    std::vector<float> logits(const SmallFeatures &features) const;
};
struct SmallSearchConfig {
    int simulations = 250, playoutDepth = 200, treeDepth = 1024;
    double exploration = -1.0, temperature = 1.0;
    bool usePrior = false, rootGuards = true;
    double priorMix = .5, priorTemperature = 1., tieEpsilon = .02;
};
struct SmallSearchStats {
    std::uint64_t simulations = 0, treeTransitions = 0, playoutTransitions = 0;
    std::uint64_t terminalWins = 0, terminalLosses = 0, terminalEscapes = 0;
    std::uint64_t depthCuts = 0, treeCuts = 0, chanceReplays = 0, nodes = 0;
    std::uint64_t policyCalls = 0, actionScores = 0;
    std::uint64_t priorCalls = 0, drawEdges = 0, guardTransitions = 0;
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
void smallRedeterminize(BattleContext &bc, std::uint64_t nonce);
int smallRootPick(const BattleContext &root, const std::vector<std::uint32_t> &actions,
                  const std::vector<int> &visits, const std::vector<double> &values,
                  double epsilon, bool guards);
struct SmallPlayoutTrace {
    std::vector<SmallFeatures> features;
    std::vector<int> choices;
    double value;
    int outcome;
    SmallSearchStats stats;
};
SmallPlayoutTrace smallCollectPlayout(const BattleContext &root, const SmallPlayoutNet &net,
                                    const SmallSearchConfig &cfg, std::uint64_t nonce);
SmallSearchResult smallSearch(const BattleContext &root, const SmallPlayoutNet &net,
                             const SmallSearchConfig &cfg, std::uint64_t nonce,
                             const SmallPlayoutNet *prior = nullptr);
}
