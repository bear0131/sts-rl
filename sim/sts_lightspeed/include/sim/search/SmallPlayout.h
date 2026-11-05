#pragma once
#include "sim/search/Action.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sts::search {
constexpr int SMALL_STATE = 32, SMALL_ACTION = 24, SMALL_HIDDEN = 64, SMALL_EMBED = 32;
constexpr int SMALL_KEYS = 10; // Existing exported action-key width / maximum hand size.
int smallKeyCount();
struct SmallActionFeatures {
    std::uint32_t bits;
    std::array<int, SMALL_KEYS> keys{};
    int keyCount = 0;
    void addKey(int key) {
        if (keyCount == SMALL_KEYS) throw std::logic_error("too many compact action keys");
        keys[keyCount++] = key;
    }
    std::array<float, SMALL_ACTION> x{};
};
struct SmallFeatures {
    std::array<float, SMALL_STATE> state{};
    std::vector<SmallActionFeatures> actions;
};
SmallFeatures smallFeatures(const BattleContext &bc);
constexpr int SMALL_MAX_CARDS_PER_TURN = 128;
bool smallTurnCardLimitReached(const BattleContext &bc);
// Parameter order is the versioned Python export contract, not a Torch ABI.
class SmallPlayoutNet {
    std::uint64_t snapshotId_;
    std::vector<float> weights_;
    std::array<float, SMALL_STATE * SMALL_HIDDEN> stateColumns_;
    std::array<float, SMALL_HIDDEN * SMALL_EMBED> hiddenColumns_;
    std::array<float, SMALL_ACTION * SMALL_EMBED> projectionColumns_;
public:
    explicit SmallPlayoutNet(std::vector<float> weights);
    std::vector<float> logits(const SmallFeatures &features) const;
    void logits(const SmallFeatures &features, std::vector<float> &scores, const float *shift=nullptr) const;
};
struct PublicPlayoutRoot;
struct SmallSearchConfig {
    std::shared_ptr<const PublicPlayoutRoot> publicRoot;
    int simulations = 250, playoutDepth = 200, treeDepth = 1024;
    double exploration = -1.0, temperature = 1.0;
    bool usePrior = false, boss = false;
    double priorMix = .5, priorTemperature = 1.;
};
struct SmallSearchStats {
    std::uint64_t simulations = 0, treeTransitions = 0, playoutTransitions = 0;
    std::uint64_t terminalWins = 0, terminalLosses = 0, terminalEscapes = 0;
    std::uint64_t depthCuts = 0, treeCuts = 0, chanceReplays = 0, nodes = 0;
    std::uint64_t policyCalls = 0, actionScores = 0;
    std::uint64_t forcedPolicySkips = 0;
    std::uint64_t priorCalls = 0, drawEdges = 0, guardTransitions = 0;
    std::uint64_t turnCardLimitDecisions = 0, turnCardLimitSimulated = 0;
    std::array<std::uint64_t, 201> playoutLengths{};
    double seconds = 0;
};
struct SmallSearchResult {
    std::uint32_t action;
    SmallFeatures features;
    std::vector<int> visits;
    std::vector<int> terminalVisits;
    std::vector<double> values;
    SmallSearchStats stats;
};
// Undefined for unfinished simulations: they are censored, not losses.
double smallTerminalValue(const BattleContext &bc, bool boss);
void smallRedeterminize(BattleContext &bc, std::uint64_t nonce);
int smallRootPick(const std::vector<int> &visits, const std::vector<double> &values,
                  std::uint64_t nonce);
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
