#ifndef STS_LIGHTSPEED_NOSL_COMBAT_TRAINER_H
#define STS_LIGHTSPEED_NOSL_COMBAT_TRAINER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
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
    bool jointPolicy = false;
    bool autoregressivePolicy = false;
    // False reproduces the pre-POL01 slot actions and ordered AR futures.
    bool mergeEquivalentActions = true;
    int selectionPrefix = 0;
    // Treat the state immediately after executing End Turn as a value-only
    // leaf.  This avoids searching through hidden enemy/random transitions.
    bool endTurnAsLeaf = false;
    std::uint64_t seed = 1;
    double dirichletAlpha = 0.0;
    double dirichletEpsilon = 0.0;
    // Root allocation mode: 0 = regular PUCT, 1 = legacy Gumbel sequential
    // halving, 2 = Full Gumbel AlphaZero/MuZero.  Mode 2 uses the completed-Q
    // transform from DeepMind's mctx implementation and deterministic
    // probability-minus-visit selection at interior nodes.
    int rootSearchMode = 0;
    // Select the canonical paper Gumbel AlphaZero path while retaining the
    // existing Full-Gumbel implementation as the default for old runs.  The
    // strict path disables project-specific coverage/value heuristics and
    // uses explicit completed-score sequential halving. Q normalization and
    // value scale remain configurable (defaults true and 0.1).
    bool paperFullGumbel = false;
    int gumbelCandidates = 0; // 0 means all legal root actions
    // Full-Gumbel q-transform parameters.  They are ignored by modes 0/1.
    double gumbelScale = 1.0;
    double gumbelValueScale = 0.1;
    double gumbelMaxVisitInit = 50.0;
    bool gumbelRescaleQ = true;
    bool gumbelUseMixedValue = true;
    // Interior Full-Gumbel exploration smoothing.  Zero preserves the
    // paper-style probability-minus-visit rule; positive values gradually
    // mix the target distribution toward uniform as node visits accumulate.
    double gumbelInteriorUniformity = 0.0;
    // Prior-independent protection: every interior edge is evaluated at least
    // this many times before the completed-policy selector is allowed to
    // concentrate visits.  One visit is enough to prevent a zero-prior edge
    // from remaining permanently unexpanded.
    int gumbelInteriorMinVisits = 1;
    // Prior-independent dynamic coverage.  A value of 0 disables it; 1.0
    // forces uniform coverage, while 0.5 guarantees half of the uniform
    // allocation at the current node visit count.
    double gumbelInteriorCoverageFraction = 0.0;
    // Decision-node value aggregation: 0=mean, 1=max, 2=progressive
    // mean->max.  Random/world-line children are always chance-averaged
    // before this setting is applied.
    int gumbelInteriorValueMode = 2;
    double gumbelInteriorMaxMixVisits = 64.0;
    // Sample hidden randomness once per decision and reuse that future for
    // every simulation.  False preserves the usual independent world lines.
    bool fixedWorldline = false;
    // Optional boss-transition terminal objective.  Zero preserves the
    // historical combat-exit HP label.  Mode 1 labels a victory with full
    // post-transition max HP; mode 2 applies the A5+ Act 1/2 restoration of
    // 75% of missing HP.  Both modes label a combat loss as zero.
    int bossTransitionValueMode = 0;
    double bossWinWeight = 0.0;
    // Experimental: send a combat victory to the same value network used
    // for nonterminal leaves. Losses retain their exact zero value.
    bool modelVictoryTerminal = false;
    // Finite unbounded outer return. Every combat exit is evaluated by the
    // caller after settling its complete sampled run continuation.
    bool outerReturn = false;
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
        int valueVisits = 0;
        double prior = 0.0;
        double value = 0.0;
        // `value` is the structurally propagated Q.  The sample mean is kept
        // separately so the inspector can distinguish a decision backup from
        // the underlying leaf samples.
        double sampleQ = 0.0;
        double rawQ = 0.0;
        double normalizedQ = 0.0;
        double completedQ = 0.0;
        double gumbel = 0.0;
        double gumbelScore = 0.0;
        double completedScore = 0.0;
        // Exact model output for an action whose immediate successor is a
        // value-only leaf (currently End Turn with endTurnAsLeaf enabled).
        // Keep this separate from propagated Q so inspectors do not mistake
        // the parent state's value or a search backup for the successor's
        // direct network evaluation.
        bool successorModelValueValid = false;
        double successorModelValue = 0.0;
        int successorModelValueCount = 0;
        bool candidate = false;
        bool activeCandidate = false;
        // True when this action has split into multiple NOSL public states.
        // Such children are a chance/world-line node and are averaged;
        // action alternatives at the parent are aggregated by the configured
        // Interior V mode.
        bool randomSplit = false;
        bool expandable = false;
        bool targeted = false;
        std::string label;
    };

    struct TreeSnapshot {
        bool valid = false;
        int visits = 0;
        int valueVisits = 0;
        double value = 0.0;
        double sampleValue = 0.0;
        // The exact finite value emitted when this public state was first
        // sent through the model. Unlike `value` and `sampleValue`, this is
        // neither clamped nor changed by search backup/action aggregation.
        bool networkValueValid = false;
        double networkValue = 0.0;
        std::vector<TreeEdgeInfo> children;
    };

    struct LeafRequest {
        std::uint64_t requestId = 0;
        // Non-owning view into the pending entry. The pointer is valid until
        // the matching backprop/abort; Python bindings serialize it before
        // returning from nextBatch().
        const BattleContext *state = nullptr;
        int selectionPrefix = 0;
        bool autoregressivePolicy = false;
        std::vector<Action> legalActions;
        bool needsPolicy = true;
        bool outerReturn = false;
        bool combatExit = false;
    };

    NoslNeuralSearchSession(const BattleContext &rootState,
                            const std::vector<double> &rootPolicyLogits,
                            NoslCombatSearchConfig config,
                            std::uint64_t searchNonce,
                            double rootValue);

    std::vector<LeafRequest> nextBatch(int maxLeaves);
    void backprop(std::uint64_t requestId, double value,
                  const std::vector<double> &policyLogits);
    void backpropBatch(
        const std::vector<std::uint64_t> &requestIds,
        const std::vector<double> &values,
        const std::vector<std::vector<double>> &policyLogits);
    // Consume contiguous Python/NumPy storage without materializing one
    // std::vector<double> per leaf. Rows are backed up in input order.
    void backpropBatchArray(const std::uint64_t *requestIds,
                            const double *values,
                            const double *policyLogits,
                            std::size_t count,
                            std::size_t policyWidth);
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
    struct ActionLabel {
        CardInstance card;
        const char *targetName = nullptr;
        std::string eager;
        bool deferred = false;
    };
    struct Edge {
        Action action;
        std::vector<Action> playAliases;
        // NOSL samples can reach the same action prefix with different public
        // observations. Keep one child node per public-state key instead of
        // merging all hidden-world outcomes into a single node.
        std::unordered_map<std::uint64_t, std::unique_ptr<Node>> children;
        int visits = 0;
        int valueVisits = 0;
        int virtualVisits = 0;
        double valueSum = 0.0;
        double prior = 0.0;
        double logPrior = 0.0;
        double propagatedValue = 0.0;
        double successorModelValueSum = 0.0;
        int successorModelValueCount = 0;
        bool propagatedValid = false;
        std::uint64_t semanticKey = 0;
        bool targeted = false;
        ActionLabel label;
        Edge(Action a, double p, bool targeted, ActionLabel label);
        Edge(Edge &&) noexcept = default;
        Edge &operator=(Edge &&) noexcept = default;
        Edge(const Edge &) = delete;
        Edge &operator=(const Edge &) = delete;
    };

    struct Node {
        int visits = 0;
        int valueVisits = 0;
        int virtualVisits = 0;
        double valueSum = 0.0;
        double modelValue = 0.0;
        double networkValue = 0.0;
        double propagatedValue = 0.0;
        bool propagatedValid = false;
        bool modelValueValid = false;
        bool networkValueValid = false;
        bool expansionPending = false;
        std::vector<Edge> edges;
    };

    struct Pending {
        std::uint64_t id = 0;
        std::unique_ptr<BattleContext> state;
        int selectionPrefix = 0;
        std::vector<Node *> nodes;
        std::vector<Edge *> edges;
        std::vector<Action> legalActions;
        Node *leaf = nullptr;
        bool needsPolicy = true;
        // The final traversed edge directly produced the value-only state
        // sent to the model.  This is distinct from an ordinary expanded
        // child node and is used to retain End Turn boundary evaluations.
        bool directSuccessorValueLeaf = false;
        // Coverage-protection simulations are retained in N for scheduling,
        // but excluded from value statistics so forced exploratory probes do
        // not bias Q/V downward.
        bool excludeValue = false;
    };

    NoslCombatSearchConfig config_;
    BattleContext rootState_;
    std::uint64_t searchNonce_ = 0;
    Node root_;
    int simulationsStarted_ = 0;
    int simulationsCompleted_ = 0;
    std::uint64_t nextRequestId_ = 1;
    std::unordered_map<std::uint64_t, Pending> pending_;
    std::vector<std::tuple<int, int, int, int, int, int, int, int, int>> stateKeyScratch_;
    // Sessions are sequential; these buffers never outlive the owning tree.
    mutable std::vector<double> rawQScratch_;
    mutable std::vector<double> scaledQScratch_;
    mutable std::vector<double> rootScoreScratch_;
    mutable std::vector<double> interiorLogitScratch_;
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
    double objectiveValue(double value) const;
    std::vector<Action> legalActions(const BattleContext &bc, int selected = 0) const;
    bool validAction(const Action &action, const BattleContext &bc, int selected) const;
    static int thinActionId(const Action &action, const BattleContext &state);
    void expandWithLogits(Node &node, const BattleContext &state,
                          const std::vector<Action> &actions,
                          const std::vector<double> &logits);
    void expandWithLogits(Node &node, const BattleContext &state,
                          const std::vector<Action> &actions,
                          const double *logits, std::size_t logitsSize);
    void applyRootDirichletNoise();
    void mergeEquivalentPlays(Node &node, const BattleContext &state);
    static void expandUniform(Node &node, const BattleContext &state,
                              const std::vector<Action> &actions);
    static bool actionTargetsEnemy(const BattleContext &state,
                                   const Action &action);
    static ActionLabel actionLabel(const BattleContext &state,
                                   const Action &action);
    static std::string formatActionLabel(const Edge &edge);
    int selectPuct(const Node &node) const;
    int selectFullGumbelInterior(const Node &node,
                                 bool *usedCoverage = nullptr) const;
    void initializeGumbelRoot();
    void initializeFullGumbelSchedule();
    struct QTransformDetails {
        std::vector<double> raw;
        std::vector<double> normalized;
        std::vector<double> scaled;
    };
    QTransformDetails completedQDetails(const Node &node) const;
    void fillCompletedQ(const Node &node, std::vector<double> &scaled,
                        std::vector<double> *raw = nullptr,
                        std::vector<double> *normalized = nullptr) const;
    const std::vector<double> &completedQTransform(const Node &node) const;
    const std::vector<double> &rootCompletedScores(bool includeNoise) const;
    std::vector<double> rootPolicyTarget() const;
    // Recompute the structural value after a leaf backup.  An Edge is a
    // chance/world-line aggregation over its public-state children; a Node
    // is a decision aggregation over its outgoing action edges.
    void recomputeEdgeValue(Edge &edge) const;
    void recomputeNodeValue(Node &node) const;
    double edgeValueForSelection(const Edge &edge,
                                 const Node &parent) const;
    double nodeValueForSelection(const Node &node) const;
    void advanceGumbelRound();
    int selectRootEdge();
    void backupStats(Pending &pending, double value);
    void recomputePathValues(const Pending &pending);
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

    static std::uint64_t stateKey(const BattleContext &bc);
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
