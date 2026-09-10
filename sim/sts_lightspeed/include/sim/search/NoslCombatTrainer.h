#ifndef STS_LIGHTSPEED_NOSL_COMBAT_TRAINER_H
#define STS_LIGHTSPEED_NOSL_COMBAT_TRAINER_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include "sim/search/Action.h"

namespace sts::search {

struct NoslCombatSearchConfig {
    int simulations = 128;
    int maxDepth = 3;
    double cPuct = 1.5;
    double fpuReduction = 0.01;
    double priorTemperature = 1.0;
    std::uint64_t seed = 1;
    double dirichletAlpha = 0.0;
    double dirichletEpsilon = 0.0;
    // Root allocation mode: 0 = regular PUCT, 1 = legacy Gumbel sequential
    // halving, 2 = Full Gumbel AlphaZero/MuZero.  Mode 2 uses the completed-Q
    // transform from DeepMind's mctx implementation and deterministic
    // probability-minus-visit selection at interior nodes.
    int rootSearchMode = 0;
    int gumbelCandidates = 0; // 0 means all legal root actions
    // Full-Gumbel q-transform parameters.  They are ignored by modes 0/1.
    double gumbelScale = 1.0;
    double gumbelValueScale = 0.1;
    double gumbelMaxVisitInit = 50.0;
    bool gumbelRescaleQ = true;
    bool gumbelUseMixedValue = true;
};

struct NoslCombatSearchStats {
    std::uint64_t simulations = 0;
    std::uint64_t transitions = 0;
    std::uint64_t expandedNodes = 0;
};

struct NoslCombatDecision {
    Action action;
    std::uint64_t stateKey = 0;
    std::vector<std::pair<std::uint32_t, int>> rootVisits;
    // Improved root policy produced by Full Gumbel's completed-Q transform.
    // Empty for legacy modes; probabilities are keyed by complete action bits
    // so target variants remain distinguishable until the Python adapter
    // aggregates them into the thin action space.
    std::vector<std::pair<std::uint32_t, double>> rootPolicy;
    int totalRootVisits = 0;
    double rootValue = 0.0;
    // Complete root action metadata.  A thin action id only identifies the
    // card slot; these fields preserve the selected enemy target for UI and
    // replay consumers.
    int selectedTarget = -1;
    bool selectedActionTargeted = false;
    bool usedFallback = false;
};

struct NoslCombatEpisodeStats {
    int decisions = 0;
    int finalHp = 0;
    int maxHp = 0;
    Outcome outcome = Outcome::UNDECIDED;
    int fallbackActions = 0;
    NoslCombatSearchStats search;
};

// A neural-guided, information-set AlphaZero session.  The simulator owns
// the tree and hidden-state clones; the caller supplies policy logits/value
// for a batch of public leaf states.  A session is deliberately incremental so
// a worker can return several leaves, let the learner run one GPU batch, and
// install all priors/values with one IPC round.
class NoslNeuralSearchSession {
public:
    struct TreeEdgeInfo {
        int edgeIndex = -1;
        std::uint32_t actionBits = 0;
        int visits = 0;
        int childVisits = 0;
        int childStates = 0;
        double prior = 0.0;
        double value = 0.0;
        // Raw/completed values are kept separately so the tree inspector can
        // explain why a sparse edge was (or was not) selected.  `value` is
        // retained as the historical raw empirical Q field.
        double rawQ = 0.0;
        double normalizedQ = 0.0;
        double completedQ = 0.0;
        double gumbel = 0.0;
        double gumbelScore = 0.0;
        double completedScore = 0.0;
        bool candidate = false;
        bool activeCandidate = false;
        bool expandable = false;
        bool targeted = false;
        std::string label;
    };

    struct TreeSnapshot {
        bool valid = false;
        int visits = 0;
        double value = 0.0;
        std::vector<TreeEdgeInfo> children;
    };

    struct LeafRequest {
        std::uint64_t requestId = 0;
        BattleContext state;
        std::vector<Action> legalActions;
        bool needsPolicy = true;
    };

    NoslNeuralSearchSession(const BattleContext &rootState,
                            const std::vector<double> &rootPolicyLogits,
                            NoslCombatSearchConfig config,
                            std::uint64_t searchNonce,
                            double rootValue);

    std::vector<LeafRequest> nextBatch(int maxLeaves);
    void backprop(std::uint64_t requestId, double value,
                  const std::vector<double> &policyLogits);
    bool done() const { return simulationsCompleted_ >= config_.simulations; }
    int simulationsCompleted() const { return simulationsCompleted_; }
    NoslCombatDecision result() const;
    // Return one node's outgoing edges.  `path` contains edge indexes from
    // root; when an edge has several public-state children, the most-visited
    // child is used as the representative for this legacy path interface.
    TreeSnapshot treeSnapshot(const std::vector<int> &path,
                              int maxChildren = 64) const;

private:
    struct Node;
    struct Edge {
        Action action;
        // NOSL samples can reach the same action prefix with different public
        // observations. Keep one child node per public-state key instead of
        // merging all hidden-world outcomes into a single node.
        std::unordered_map<std::uint64_t, std::unique_ptr<Node>> children;
        int visits = 0;
        int virtualVisits = 0;
        double valueSum = 0.0;
        double prior = 0.0;
        std::uint64_t semanticKey = 0;
        bool targeted = false;
        std::string label;
        Edge(Action a, double p, bool targeted, std::string label)
            : action(a), prior(p), targeted(targeted), label(std::move(label)) {}
        Edge(Edge &&) noexcept = default;
        Edge &operator=(Edge &&) noexcept = default;
        Edge(const Edge &) = delete;
        Edge &operator=(const Edge &) = delete;
    };

    struct Node {
        int visits = 0;
        int virtualVisits = 0;
        double valueSum = 0.0;
        double networkValue = 0.0;
        bool expansionPending = false;
        std::vector<Edge> edges;
    };

    struct Pending {
        std::uint64_t id = 0;
        BattleContext state;
        std::vector<Node *> nodes;
        std::vector<Edge *> edges;
        std::vector<Action> legalActions;
        Node *leaf = nullptr;
        bool needsPolicy = true;
    };

    NoslCombatSearchConfig config_;
    BattleContext rootState_;
    std::uint64_t searchNonce_ = 0;
    Node root_;
    int simulationsStarted_ = 0;
    int simulationsCompleted_ = 0;
    std::uint64_t nextRequestId_ = 1;
    std::unordered_map<std::uint64_t, Pending> pending_;
    std::vector<int> gumbelActiveEdges_;
    // Initial Gumbel-Top-k set.  `gumbelActiveEdges_` is narrowed after each
    // Sequential-Halving round, so retaining both sets is useful for audit/UI.
    std::vector<int> gumbelInitialEdges_;
    // Standard-Gumbel noise is kept separately from log(prior).  The legacy
    // implementation stores their sum in gumbelScores_; Full Gumbel needs the
    // individual noise for its final score and policy target.
    std::vector<double> gumbelNoise_;
    std::vector<double> gumbelScores_;
    // mctx's sequence of considered visit counts.  Entry i specifies the
    // child visit count eligible for root simulation i.
    std::vector<int> gumbelVisitSchedule_;
    int gumbelRound_ = 0;
    int gumbelRoundQuota_ = 0;
    int gumbelRoundStarted_ = 0;
    int gumbelRoundCompleted_ = 0;
    int gumbelRemaining_ = 0;

    static double hpValue(const BattleContext &bc);
    static std::vector<Action> legalActions(const BattleContext &bc);
    static int thinActionId(const Action &action);
    void expandWithLogits(Node &node, const BattleContext &state,
                          const std::vector<Action> &actions,
                          const std::vector<double> &logits);
    void applyRootDirichletNoise();
    static void expandUniform(Node &node, const BattleContext &state,
                              const std::vector<Action> &actions);
    static bool actionTargetsEnemy(const BattleContext &state,
                                   const Action &action);
    static std::string actionLabel(const BattleContext &state,
                                   const Action &action);
    int selectPuct(const Node &node) const;
    int selectFullGumbelInterior(const Node &node) const;
    void initializeGumbelRoot();
    void initializeFullGumbelSchedule();
    struct QTransformDetails {
        std::vector<double> raw;
        std::vector<double> normalized;
        std::vector<double> scaled;
    };
    QTransformDetails completedQDetails(const Node &node) const;
    std::vector<double> completedQTransform(const Node &node) const;
    std::vector<double> rootCompletedScores(bool includeNoise) const;
    std::vector<double> rootPolicyTarget() const;
    void advanceGumbelRound();
    int selectRootEdge();
    void backup(Pending &pending, double value);
    void finishTerminal(Pending &pending, double value);
};

// Native combat-only search/training prototype. It intentionally has no
// neural-network dependency: the tabular policy/value are used to validate
// NOSL semantics and simulator throughput before wiring in a value network.
class NoslCombatTrainer {
public:
    explicit NoslCombatTrainer(NoslCombatSearchConfig config = {});

    NoslCombatDecision search(const BattleContext &rootState,
                              std::uint64_t searchNonce,
                              NoslCombatSearchStats *stats = nullptr);

    NoslCombatEpisodeStats runEpisode(std::uint64_t seed,
                                      CharacterClass character = CharacterClass::IRONCLAD,
                                      int ascension = 0,
                                      MonsterEncounter encounter = MonsterEncounter::JAW_WORM,
                                      int maxDecisions = 500,
                                      bool train = true);

    void clearTable();
    bool saveTable(const std::string &path) const;
    std::size_t policyStateCount() const { return policy_.size(); }
    std::size_t valueStateCount() const { return values_.size(); }

private:
    struct Node;
    struct Edge {
        Action action;
        std::unordered_map<std::uint64_t, std::unique_ptr<Node>> children;
        int visits = 0;
        double valueSum = 0.0;
        double prior = 0.0;
        std::uint64_t semanticKey = 0;

        Edge(Action a, double p) : action(a), prior(p) {}
        Edge(Edge &&) noexcept = default;
        Edge &operator=(Edge &&) noexcept = default;
        Edge(const Edge &) = delete;
        Edge &operator=(const Edge &) = delete;
    };

    struct Node {
        int visits = 0;
        double valueSum = 0.0;
        std::vector<Edge> edges;
    };

    struct PolicyState {
        std::unordered_map<std::uint32_t, double> probabilities;
    };

    NoslCombatSearchConfig config_;
    std::unordered_map<std::uint64_t, PolicyState> policy_;
    std::unordered_map<std::uint64_t, double> values_;

    static std::uint64_t stateKey(const BattleContext &bc);
    static double hpValue(const BattleContext &bc);
    static std::vector<Action> legalActions(const BattleContext &bc);

    void expandNode(Node &node, const BattleContext &state,
                    std::uint64_t key,
                    const std::vector<Action> &actions) const;
    int selectPuct(const Node &node, const BattleContext &state) const;
    double leafValue(const BattleContext &state) const;
    static Action firstLegalAction(const BattleContext &bc);
    void updateFromDecision(const NoslCombatDecision &decision, double target);
};

}

#endif // STS_LIGHTSPEED_NOSL_COMBAT_TRAINER_H
