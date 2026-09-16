#include "sim/search/NoslCombatTrainer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <numeric>

#include "constants/MonsterMoves.h"
#include "sim/search/Nosl.h"

namespace sts::search {
namespace {

// Keep native terminal/truncation targets on the same scale as the Python
// learner's HP target (encode.HP_SCALE).
constexpr double HP_VALUE_SCALE = 200.0;

inline void hashCombine(std::uint64_t &hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

template <typename T>
inline void hashValue(std::uint64_t &hash, T value) {
    hashCombine(hash, static_cast<std::uint64_t>(value));
}

template <typename Pile>
void hashCardComposition(std::uint64_t &hash, const Pile &pile) {
    // Card id/upgrade count is not a complete card state.  Temporary cards
    // and cards such as Masterful Stab/Blood for Blood carry transition
    // relevant data in specialData, while Snecko/free-play/retain effects
    // are carried by the other per-instance fields.  Keep the collection
    // unordered for NOSL, but retain every field that can affect a future
    // transition.  uniqueId is deliberately excluded: it is an opaque card
    // identity and is not public information.
    using CardSignature = std::tuple<int, int, int, int, int, int, int>;
    // CardManager uses std::vector unless the optional fixed-list build flag
    // is enabled, so a legal pile is not necessarily bounded by
    // MAX_GROUP_SIZE.  In particular, long runs and generated cards can make
    // a pile exceed 64 entries.  Size this scratch space from the actual pile
    // to avoid writing past a fixed stack array while canonicalizing the key.
    std::vector<CardSignature> cards;
    cards.reserve(pile.size());
    for (const auto &card : pile) {
        cards.emplace_back(
            static_cast<int>(card.getId()),
            static_cast<int>(card.specialData),
            static_cast<int>(card.cost),
            static_cast<int>(card.costForTurn),
            card.upgraded ? 1 : 0,
            card.freeToPlayOnce ? 1 : 0,
            card.retain ? 1 : 0);
    }
    std::sort(cards.begin(), cards.end());
    hashValue(hash, cards.size());
    for (const auto &[cardId, specialData, cost, costForTurn, upgraded,
                       freeToPlayOnce, retain] : cards) {
        hashValue(hash, cardId);
        hashValue(hash, specialData);
        hashValue(hash, cost);
        hashValue(hash, costForTurn);
        hashValue(hash, upgraded);
        hashValue(hash, freeToPlayOnce);
        hashValue(hash, retain);
    }
}

void hashHandComposition(std::uint64_t &hash, const BattleContext &bc) {
    using CardSignature = std::tuple<int, int, int, int, int, int, int, int>;
    std::array<CardSignature, CardManager::MAX_HAND_SIZE> cards{};
    std::size_t count = 0;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto &card = bc.cards.hand[i];
        const int currentCost = card.freeToPlayOnce ? 0 : card.costForTurn;
        cards[count++] = CardSignature(
            static_cast<int>(card.getId()),
            static_cast<int>(card.specialData),
            static_cast<int>(card.cost),
            static_cast<int>(card.costForTurn),
            currentCost,
            card.upgraded ? 1 : 0,
            card.freeToPlayOnce ? 1 : 0,
            card.retain ? 1 : 0);
    }
    std::sort(cards.begin(), cards.begin() + count);
    hashValue(hash, count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto &[cardId, specialData, cost, costForTurn, currentCost,
                     upgraded, freeToPlayOnce, retain] = cards[i];
        hashValue(hash, cardId);
        hashValue(hash, specialData);
        hashValue(hash, cost);
        hashValue(hash, costForTurn);
        hashValue(hash, currentCost);
        hashValue(hash, upgraded);
        hashValue(hash, freeToPlayOnce);
        hashValue(hash, retain);
    }
}

// Hash the public/effective combat state that determines the next observable
// transition.  The simulator has a few compact fields (for example stance,
// just-applied bits, and monster unique powers) that are not emitted as
// standalone raw fields but still change the next legal transition, so they
// must be part of the information-set key.  Hidden RNG streams, pile order,
// and hand slot order remain deliberately excluded for NOSL.
std::uint64_t publicStateKey(const BattleContext &bc) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    hashValue(hash, static_cast<int>(bc.inputState));
    hashValue(hash, static_cast<int>(bc.outcome));
    hashValue(hash, bc.turn);
    hashValue(hash, bc.floorNum);
    hashValue(hash, static_cast<int>(bc.encounter));
    hashValue(hash, bc.ascension);
    hashValue(hash, bc.monsterTurnIdx);
    hashValue(hash, bc.isBattleOver ? 1 : 0);
    hashValue(hash, bc.endTurnQueued ? 1 : 0);
    hashValue(hash, bc.turnHasEnded ? 1 : 0);
    hashValue(hash, bc.skipMonsterTurn ? 1 : 0);
    hashValue(hash, bc.postEnemyTurnBoundary ? 1 : 0);
    hashValue(hash, bc.miscBits.to_ulong());
    hashValue(hash, bc.loopCount);
    hashValue(hash, bc.energyWasted);
    hashValue(hash, bc.cardsDrawn);
    hashValue(hash, bc.haveUsedDiscoveryAction ? 1 : 0);
    hashValue(hash, bc.undefinedBehaviorEvoked ? 1 : 0);

    const auto &player = bc.player;
    hashValue(hash, static_cast<int>(player.cc));
    hashValue(hash, player.curHp);
    hashValue(hash, player.maxHp);
    hashValue(hash, player.block);
    hashValue(hash, player.energy);
    hashValue(hash, player.energyPerTurn);
    hashValue(hash, player.cardDrawPerTurn);
    hashValue(hash, player.gold);
    hashValue(hash, static_cast<int>(player.stance));
    hashValue(hash, player.orbSlots);
    hashValue(hash, player.lastTargetedMonster);
    hashValue(hash, player.artifact);
    hashValue(hash, player.dexterity);
    hashValue(hash, player.focus);
    hashValue(hash, player.strength);
    hashValue(hash, player.justAppliedBits);
    hashValue(hash, player.statusBits0);
    hashValue(hash, player.statusBits1);
    hashValue(hash, player.relicBits0);
    hashValue(hash, player.relicBits1);
    hashValue(hash, player.happyFlowerCounter);
    hashValue(hash, player.incenseBurnerCounter);
    hashValue(hash, player.inkBottleCounter);
    hashValue(hash, player.inserterCounter);
    hashValue(hash, player.nunchakuCounter);
    hashValue(hash, player.penNibCounter);
    hashValue(hash, player.sundialCounter);
    hashValue(hash, player.haveUsedNecronomiconThisTurn ? 1 : 0);
    hashValue(hash, player.combustHpLoss);
    hashValue(hash, player.devaFormEnergyPerTurn);
    hashValue(hash, player.echoFormCardsDoubled);
    hashValue(hash, player.panacheCounter);
    hashValue(hash, player.cardsPlayedThisTurn);
    hashValue(hash, player.attacksPlayedThisTurn);
    hashValue(hash, player.skillsPlayedThisTurn);
    hashValue(hash, player.orangePelletsCardTypesPlayed.to_ulong());
    hashValue(hash, player.cardsDiscardedThisTurn);
    hashValue(hash, player.timesDamagedThisCombat);
    hashValue(hash, player.bomb1);
    hashValue(hash, player.bomb2);
    hashValue(hash, player.bomb3);
    hashValue(hash, player.lastAttackUnblockedDamage);

    const int playerStatusCount = static_cast<int>(
        sizeof(playerStatusStrings) / sizeof(playerStatusStrings[0]));
    int activePlayerStatuses = 0;
    for (int i = 1; i < playerStatusCount; ++i) {
        const auto status = static_cast<PlayerStatus>(i);
        if (player.hasStatusRuntime(status)) {
            ++activePlayerStatuses;
        }
    }
    hashValue(hash, activePlayerStatuses);
    for (int i = 1; i < playerStatusCount; ++i) {
        const auto status = static_cast<PlayerStatus>(i);
        if (player.hasStatusRuntime(status)) {
            hashValue(hash, i);
            hashValue(hash, player.getStatusRuntime(status));
        }
    }

    hashHandComposition(hash, bc);
    hashCardComposition(hash, bc.cards.drawPile);
    hashCardComposition(hash, bc.cards.discardPile);
    hashCardComposition(hash, bc.cards.exhaustPile);
    hashCardComposition(hash, bc.cards.limbo);
    hashCardComposition(hash, bc.cards.stasisCards);
    hashValue(hash, bc.cards.cardsInHand);
    hashValue(hash, bc.cards.handNormalityCount);
    hashValue(hash, bc.cards.handPainCount);
    hashValue(hash, bc.cards.strikeCount);
    hashValue(hash, bc.cards.handBloodCardCount);
    hashValue(hash, bc.cards.drawPileBloodCardCount);
    hashValue(hash, bc.cards.discardPileBloodCardCount);

    hashValue(hash, bc.potionCapacity);
    hashValue(hash, bc.potionCount);
    for (int i = 0; i < bc.potionCapacity; ++i) {
        hashValue(hash, static_cast<int>(bc.potions[i]));
    }

    hashValue(hash, static_cast<int>(bc.cardSelectInfo.cardSelectTask));
    hashValue(hash, bc.cardSelectInfo.pickCount);
    hashValue(hash, bc.cardSelectInfo.canPickZero ? 1 : 0);
    hashValue(hash, bc.cardSelectInfo.canPickAnyNumber ? 1 : 0);
    hashValue(hash, bc.cardSelectInfo.data0);
    for (const auto card : bc.cardSelectInfo.cards) {
        hashValue(hash, static_cast<int>(card));
    }

    hashValue(hash, bc.monsters.monsterCount);
    hashValue(hash, bc.monsters.monstersAlive);
    hashValue(hash, bc.monsters.extraRollMoveOnTurn.to_ulong());
    hashValue(hash, bc.monsters.skipTurn.to_ulong());
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &monster = bc.monsters.arr[i];
        hashValue(hash, monster.idx);
        hashValue(hash, static_cast<int>(monster.id));
        hashValue(hash, monster.curHp);
        hashValue(hash, monster.maxHp);
        hashValue(hash, monster.block);
        hashValue(hash, monster.isDeadOrEscaped() ? 1 : 0);
        hashValue(hash, monster.halfDead ? 1 : 0);
        hashValue(hash, monster.isEscapingB ? 1 : 0);
        hashValue(hash, monster.escapeNext ? 1 : 0);
        hashValue(hash, static_cast<int>(monster.moveHistory[0]));
        hashValue(hash, static_cast<int>(monster.moveHistory[1]));
        hashValue(hash, monster.statusBits);
        hashValue(hash, monster.artifact);
        hashValue(hash, monster.blockReturn);
        hashValue(hash, monster.choked);
        hashValue(hash, monster.corpseExplosion);
        hashValue(hash, monster.lockOn);
        hashValue(hash, monster.mark);
        hashValue(hash, monster.metallicize);
        hashValue(hash, monster.platedArmor);
        hashValue(hash, monster.poison);
        hashValue(hash, monster.regen);
        hashValue(hash, monster.shackled);
        hashValue(hash, monster.strength);
        hashValue(hash, monster.vulnerable);
        hashValue(hash, monster.weak);
        hashValue(hash, monster.uniquePower0);
        hashValue(hash, monster.uniquePower1);
        hashValue(hash, monster.miscInfo);
        const auto damage = monster.getMoveBaseDamage(bc);
        hashValue(hash, damage.damage);
        hashValue(hash, damage.attackCount);
        hashValue(hash, damage.damage > 0
            ? monster.calculateDamageToPlayer(bc, damage.damage)
            : 0);
        for (int statusId = 0;
             statusId < static_cast<int>(MonsterStatus::INVALID); ++statusId) {
            const auto status = static_cast<MonsterStatus>(statusId);
            const int value = monster.getStatusInternal(status);
            if (value != 0) {
                hashValue(hash, statusId);
                hashValue(hash, value);
            }
        }
        // Delimit each monster's status list so a changed slot cannot be
        // confused with a changed status value in the next slot.
        hashValue(hash, static_cast<int>(0x4D53544D));
    }
    return hash;
}

enum class ActionCardCollection : int {
    HAND = 1,
    DRAW,
    DISCARD,
    EXHAUST,
};

int cardSelectionCollection(CardSelectTask task) {
    switch (task) {
        case CardSelectTask::ARMAMENTS:
        case CardSelectTask::DUAL_WIELD:
        case CardSelectTask::EXHAUST_ONE:
        case CardSelectTask::FORETHOUGHT:
        case CardSelectTask::NIGHTMARE:
        case CardSelectTask::RECYCLE:
        case CardSelectTask::SETUP:
        case CardSelectTask::WARCRY:
            return static_cast<int>(ActionCardCollection::HAND);
        case CardSelectTask::EXHUME:
            return static_cast<int>(ActionCardCollection::EXHAUST);
        case CardSelectTask::HEADBUTT:
        case CardSelectTask::HOLOGRAM:
        case CardSelectTask::LIQUID_MEMORIES_POTION:
        case CardSelectTask::MEDITATE:
            return static_cast<int>(ActionCardCollection::DISCARD);
        case CardSelectTask::SECRET_TECHNIQUE:
        case CardSelectTask::SECRET_WEAPON:
        case CardSelectTask::SEEK:
            return static_cast<int>(ActionCardCollection::DRAW);
        default:
            return 0;
    }
}

void hashActionCard(std::uint64_t &hash, const CardInstance &card,
                    bool includeCost) {
    hashValue(hash, static_cast<int>(card.getId()));
    hashValue(hash, static_cast<int>(card.specialData));
    hashValue(hash, static_cast<int>(card.cost));
    hashValue(hash, static_cast<int>(card.costForTurn));
    hashValue(hash, card.upgraded ? 1 : 0);
    hashValue(hash, card.freeToPlayOnce ? 1 : 0);
    hashValue(hash, card.retain ? 1 : 0);
    if (includeCost) {
        hashValue(hash, card.freeToPlayOnce ? 0 : card.costForTurn);
    }
}

const CardInstance *actionCardAt(const BattleContext &bc, int collection,
                                 int index) {
    switch (static_cast<ActionCardCollection>(collection)) {
        case ActionCardCollection::HAND:
            return index >= 0 && index < bc.cards.cardsInHand
                ? &bc.cards.hand[index] : nullptr;
        case ActionCardCollection::DRAW:
            return index >= 0 && index < static_cast<int>(bc.cards.drawPile.size())
                ? &bc.cards.drawPile[index] : nullptr;
        case ActionCardCollection::DISCARD:
            return index >= 0 && index < static_cast<int>(bc.cards.discardPile.size())
                ? &bc.cards.discardPile[index] : nullptr;
        case ActionCardCollection::EXHAUST:
            return index >= 0 && index < static_cast<int>(bc.cards.exhaustPile.size())
                ? &bc.cards.exhaustPile[index] : nullptr;
        default:
            return nullptr;
    }
}

// Action ids use hand/pile indexes, but those indexes are not stable across
// NOSL samples when the corresponding collection is treated as a composition.
// This key lets a selected edge be remapped to the equivalent current index.
std::uint64_t publicActionKey(const BattleContext &bc, const Action &action) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    hashValue(hash, static_cast<int>(action.getActionType()));
    switch (action.getActionType()) {
        case ActionType::CARD: {
            hashValue(hash, action.getTargetIdx());
            const auto *card = actionCardAt(
                bc, static_cast<int>(ActionCardCollection::HAND),
                action.getSourceIdx());
            if (card != nullptr) {
                hashActionCard(hash, *card, true);
            } else {
                hashValue(hash, action.getSourceIdx());
            }
            break;
        }
        case ActionType::POTION:
            // Potion slots are observable and remain positional.
            hashValue(hash, action.getSourceIdx());
            hashValue(hash, action.getTargetIdx());
            if (action.getSourceIdx() >= 0 &&
                action.getSourceIdx() < bc.potionCapacity) {
                hashValue(hash, static_cast<int>(
                    bc.potions[action.getSourceIdx()]));
            }
            break;
        case ActionType::SINGLE_CARD_SELECT: {
            hashValue(hash, static_cast<int>(
                bc.cardSelectInfo.cardSelectTask));
            const int collection = cardSelectionCollection(
                bc.cardSelectInfo.cardSelectTask);
            const auto *card = actionCardAt(
                bc, collection, action.getSelectIdx());
            if (card != nullptr) {
                hashActionCard(
                    hash, *card,
                    collection == static_cast<int>(ActionCardCollection::HAND));
            } else if (bc.cardSelectInfo.cardSelectTask == CardSelectTask::CODEX ||
                       bc.cardSelectInfo.cardSelectTask == CardSelectTask::DISCOVERY) {
                const int index = action.getSelectIdx();
                if (index >= 0 && index < static_cast<int>(bc.cardSelectInfo.cards.size())) {
                    hashValue(hash, static_cast<int>(bc.cardSelectInfo.cards[index]));
                } else {
                    hashValue(hash, index);
                }
            } else {
                hashValue(hash, action.getSelectIdx());
            }
            break;
        }
        case ActionType::MULTI_CARD_SELECT: {
            hashValue(hash, static_cast<int>(
                bc.cardSelectInfo.cardSelectTask));
            std::array<std::tuple<int, int, int>, CardManager::MAX_HAND_SIZE>
                selected{};
            std::size_t selectedCount = 0;
            for (const int index : action.getSelectedIdxs()) {
                const auto *card = actionCardAt(
                    bc, static_cast<int>(ActionCardCollection::HAND), index);
                if (card != nullptr) {
                selected[selectedCount++] = std::make_tuple(
                        static_cast<int>(card->getId()),
                        card->specialData,
                        card->freeToPlayOnce ? 0 : card->costForTurn);
                } else {
                    selected[selectedCount++] = std::make_tuple(-1, index, 0);
                }
            }
            std::sort(selected.begin(), selected.begin() + selectedCount);
            hashValue(hash, selectedCount);
            for (std::size_t i = 0; i < selectedCount; ++i) {
                const auto &[cardId, upgrades, cost] = selected[i];
                hashValue(hash, cardId);
                hashValue(hash, upgrades);
                hashValue(hash, cost);
            }
            break;
        }
        case ActionType::END_TURN:
        default:
            break;
    }
    return hash;
}

}

NoslCombatTrainer::NoslCombatTrainer(NoslCombatSearchConfig config)
    : config_(config) {
    if (config_.simulations < 1) {
        config_.simulations = 1;
    }
    if (config_.maxDepth < 0) {
        config_.maxDepth = 0;
    }
    // Zero is a valid ablation value: it removes the PUCT prior term while
    // retaining Q-based selection.  Only negative/non-finite values fall back
    // to the historical default.
    if (config_.cPuct < 0.0 || !std::isfinite(config_.cPuct)) {
        config_.cPuct = 1.5;
    }
    if (config_.priorTemperature <= 0.0 ||
        !std::isfinite(config_.priorTemperature)) {
        // A zero temperature is meaningful for action sampling at the
        // Python layer, but cannot be used to turn logits into priors.
        config_.priorTemperature = 1.0;
    }
    if (config_.priorTemperature <= 0.0 ||
        !std::isfinite(config_.priorTemperature)) {
        config_.priorTemperature = 1.0;
    }
}

std::uint64_t NoslCombatTrainer::stateKey(const BattleContext &bc) {
    return publicStateKey(bc);
}

double NoslCombatTrainer::hpValue(const BattleContext &bc) {
    if (bc.player.maxHp <= 0) {
        return 0.0;
    }
    return std::clamp(
        static_cast<double>(bc.player.curHp) / HP_VALUE_SCALE, 0.0, 1.0);
}

std::vector<Action> NoslCombatTrainer::legalActions(const BattleContext &bc) {
    return Action::enumerateCombatActions(bc);
}

void NoslCombatTrainer::expandNode(Node &node, const BattleContext &state,
                                   std::uint64_t key,
                                   const std::vector<Action> &actions) const {
    node.edges.reserve(actions.size());
    const auto stateIt = policy_.find(key);
    if (stateIt == policy_.end()) {
        const double prior = 1.0 / std::max<std::size_t>(1, actions.size());
        for (const auto &action : actions) {
            node.edges.emplace_back(action, prior);
            node.edges.back().semanticKey = publicActionKey(state, action);
        }
        return;
    }

    double total = 0.0;
    std::vector<double> weights;
    weights.reserve(actions.size());
    for (const auto &action : actions) {
        const auto it = stateIt->second.probabilities.find(action.bits);
        const double value = it == stateIt->second.probabilities.end()
            ? 1.0
            : std::max(1e-6, it->second);
        weights.push_back(value);
        total += value;
    }
    for (std::size_t i = 0; i < actions.size(); ++i) {
        node.edges.emplace_back(
            actions[i],
            total > 0.0 ? weights[i] / total : 1.0 / actions.size()
        );
        node.edges.back().semanticKey = publicActionKey(state, actions[i]);
    }
}

int NoslCombatTrainer::selectPuct(const Node &node, const BattleContext &state) const {
    if (node.edges.empty()) {
        return -1;
    }
    const double parentVisits = std::sqrt(static_cast<double>(node.visits + 1));
    const double defaultValue = leafValue(state);
    int bestIndex = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(node.edges.size()); ++i) {
        const auto &edge = node.edges[i];
        const double q = edge.visits > 0
            ? edge.valueSum / edge.visits
            : defaultValue;
        const double exploration = config_.cPuct * edge.prior * parentVisits / (1 + edge.visits);
        const double score = q + exploration;
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }
    return bestIndex;
}

double NoslCombatTrainer::leafValue(const BattleContext &state) const {
    const auto it = values_.find(stateKey(state));
    if (it != values_.end() && std::isfinite(it->second)) {
        return std::clamp(it->second, 0.0, 1.0);
    }
    return hpValue(state);
}

Action NoslCombatTrainer::firstLegalAction(const BattleContext &bc) {
    const auto actions = legalActions(bc);
    return actions.empty() ? Action() : actions.front();
}

NoslCombatDecision NoslCombatTrainer::search(const BattleContext &rootState,
                                              std::uint64_t searchNonce,
                                              NoslCombatSearchStats *stats) {
    NoslCombatSearchStats localStats;
    Node root;
    const auto rootKey = stateKey(rootState);

    for (int simulation = 0; simulation < config_.simulations; ++simulation) {
        BattleContext state(rootState);
        noslRedeterminize(state, noslMix(searchNonce + static_cast<std::uint64_t>(simulation)));

        Node *node = &root;
        std::vector<Node *> nodePath{node};
        std::vector<Edge *> edgePath;
        for (int depth = 0;
             (config_.maxDepth == 0 || depth < config_.maxDepth) &&
             state.outcome == Outcome::UNDECIDED;
             ++depth) {
            if (node->edges.empty()) {
                const auto actions = legalActions(state);
                expandNode(*node, state, stateKey(state), actions);
                ++localStats.expandedNodes;
                if (node->edges.empty()) {
                    break;
                }
            }

            const int edgeIndex = selectPuct(*node, state);
            if (edgeIndex < 0) {
                break;
            }
            auto *edge = &node->edges[edgeIndex];
            Action executableAction;
            bool foundAction = false;
            if (edge->action.isValidAction(state) &&
                publicActionKey(state, edge->action) == edge->semanticKey) {
                executableAction = edge->action;
                foundAction = true;
            } else {
                // Hand/pile indexes can change between public states that
                // share the same composition key. Find the current legal
                // action carrying this edge's semantic card/target key.
                const auto currentActions = legalActions(state);
                for (const auto &candidateAction : currentActions) {
                    const auto candidateKey = publicActionKey(
                        state, candidateAction);
                    if (candidateKey != edge->semanticKey) {
                        continue;
                    }
                    for (auto &candidateEdge : node->edges) {
                        if (&candidateEdge != edge &&
                            candidateEdge.semanticKey == candidateKey) {
                            edge = &candidateEdge;
                            break;
                        }
                    }
                    executableAction = candidateAction;
                    foundAction = true;
                    break;
                }
            }
            if (!foundAction) {
                // Preserve the defensive fallback for a state whose action
                // set has no semantic counterpart in this node.
                edge = nullptr;
                for (auto &candidate : node->edges) {
                    if (candidate.action.isValidAction(state)) {
                        edge = &candidate;
                        executableAction = candidate.action;
                        foundAction = true;
                        break;
                    }
                }
                if (!foundAction) {
                    break;
                }
            }
            edgePath.push_back(edge);
            state.stopAfterEnemyTurn =
                config_.endTurnAsLeaf &&
                executableAction.getActionType() == ActionType::END_TURN;
            executableAction.execute(state);
            ++localStats.transitions;

            if (config_.endTurnAsLeaf &&
                executableAction.getActionType() == ActionType::END_TURN) {
                // executeActions stopped at the post-enemy-turn boundary.
                break;
            }

            const auto childKey = publicStateKey(state);
            auto childIt = edge->children.find(childKey);
            if (childIt == edge->children.end()) {
                childIt = edge->children.emplace(
                    childKey, std::make_unique<Node>()).first;
            }
            node = childIt->second.get();
            nodePath.push_back(node);
        }

        const double value = leafValue(state);
        for (auto *visitedNode : nodePath) {
            ++visitedNode->visits;
            visitedNode->valueSum += value;
        }
        for (auto *visitedEdge : edgePath) {
            ++visitedEdge->visits;
            visitedEdge->valueSum += value;
        }
        ++localStats.simulations;
    }

    NoslCombatDecision result;
    result.stateKey = rootKey;
    result.totalRootVisits = root.visits;
    result.rootValue = root.visits > 0
        ? root.valueSum / static_cast<double>(root.visits)
        : leafValue(rootState);
    if (root.edges.empty()) {
        result.action = firstLegalAction(rootState);
    } else {
        int bestIndex = 0;
        for (int i = 1; i < static_cast<int>(root.edges.size()); ++i) {
            if (root.edges[i].visits > root.edges[bestIndex].visits ||
                (root.edges[i].visits == root.edges[bestIndex].visits &&
                 root.edges[i].valueSum > root.edges[bestIndex].valueSum)) {
                bestIndex = i;
            }
        }
        result.action = root.edges[bestIndex].action;
        result.rootVisits.reserve(root.edges.size());
        for (const auto &edge : root.edges) {
            result.rootVisits.emplace_back(edge.action.bits, edge.visits);
        }
    }

    if (stats) {
        *stats = localStats;
    }
    return result;
}

void NoslCombatTrainer::updateFromDecision(const NoslCombatDecision &decision, double target) {
    if (decision.totalRootVisits <= 0) {
        return;
    }
    constexpr double policyLearningRate = 0.1;
    constexpr double valueLearningRate = 0.05;
    auto &state = policy_[decision.stateKey];
    for (const auto &[bits, visits] : decision.rootVisits) {
        const double desired = static_cast<double>(visits) / decision.totalRootVisits;
        auto it = state.probabilities.find(bits);
        if (it == state.probabilities.end()) {
            it = state.probabilities.emplace(bits, desired).first;
        } else {
            it->second += policyLearningRate * (desired - it->second);
        }
    }
    auto valueIt = values_.find(decision.stateKey);
    if (valueIt == values_.end()) {
        values_.emplace(decision.stateKey, target);
    } else {
        valueIt->second += valueLearningRate * (target - valueIt->second);
    }
}

NoslCombatEpisodeStats NoslCombatTrainer::runEpisode(
    std::uint64_t seed,
    CharacterClass character,
    int ascension,
    MonsterEncounter encounter,
    int maxDecisions,
    bool train) {
    GameContext game(character, seed, ascension);
    BattleContext battle;
    battle.init(game, encounter);

    NoslCombatEpisodeStats episode;
    episode.maxHp = battle.player.maxHp;
    std::vector<NoslCombatDecision> decisions;
    decisions.reserve(maxDecisions);

    for (int decisionIdx = 0;
         decisionIdx < maxDecisions && battle.outcome == Outcome::UNDECIDED;
         ++decisionIdx) {
        NoslCombatSearchStats searchStats;
        auto decision = search(battle, noslMix(seed + static_cast<std::uint64_t>(decisionIdx)), &searchStats);
        episode.search.simulations += searchStats.simulations;
        episode.search.transitions += searchStats.transitions;
        episode.search.expandedNodes += searchStats.expandedNodes;

        if (!decision.action.isValidAction(battle)) {
            decision.action = firstLegalAction(battle);
            decision.usedFallback = true;
            ++episode.fallbackActions;
        }
        if (decision.action.bits == static_cast<std::uint32_t>(-1)) {
            break;
        }
        if (train) {
            decisions.push_back(decision);
        }
        decision.action.execute(battle);
        ++episode.decisions;
    }

    episode.finalHp = battle.player.curHp;
    episode.outcome = battle.outcome;
    if (train) {
        const double target = hpValue(battle);
        for (const auto &decision : decisions) {
            updateFromDecision(decision, target);
        }
    }
    return episode;
}

// ---------------------------------------------------------------------------
// Neural-guided AlphaZero session

NoslNeuralSearchSession::NoslNeuralSearchSession(
    const BattleContext &rootState,
    const std::vector<double> &rootPolicyLogits,
    NoslCombatSearchConfig config,
    std::uint64_t searchNonce,
    double rootValue)
    : config_(config), rootState_(rootState), searchNonce_(searchNonce) {
    // NOSL treats draw-pile order as hidden. Canonicalize the copied root once
    // so every simulation can shuffle the same representation without paying
    // an O(n log n) sort before its independent permutation.
    std::sort(
        rootState_.cards.drawPile.begin(), rootState_.cards.drawPile.end(),
        [](const CardInstance &lhs, const CardInstance &rhs) {
            return lhs.getUniqueId() < rhs.getUniqueId();
        });
    if (config_.simulations < 1) {
        config_.simulations = 1;
    }
    if (config_.maxDepth < 0) {
        config_.maxDepth = 0;
    }
    if (config_.cPuct < 0.0 || !std::isfinite(config_.cPuct)) {
        config_.cPuct = 1.5;
    }
    if (config_.fpuReduction < 0.0 || !std::isfinite(config_.fpuReduction)) {
        config_.fpuReduction = 0.01;
    }
    if (!(config_.dirichletAlpha > 0.0) ||
        !std::isfinite(config_.dirichletAlpha) ||
        !(config_.dirichletEpsilon > 0.0) ||
        !std::isfinite(config_.dirichletEpsilon)) {
        config_.dirichletAlpha = 0.0;
        config_.dirichletEpsilon = 0.0;
    } else {
        config_.dirichletEpsilon = std::clamp(config_.dirichletEpsilon, 0.0, 1.0);
    }
    if (config_.rootSearchMode != 1 && config_.rootSearchMode != 2) {
        config_.rootSearchMode = 0;
    }
    if (config_.priorTemperature <= 0.0 ||
        !std::isfinite(config_.priorTemperature)) {
        // Temperature is used while converting policy logits to priors.  A
        // zero temperature is an acting-time convention, not a valid divisor
        // for tree expansion; keep the native session finite and deterministic.
        config_.priorTemperature = 1.0;
    }
    if (config_.gumbelCandidates < 0) {
        config_.gumbelCandidates = 0;
    }
    if (config_.gumbelScale < 0.0 || !std::isfinite(config_.gumbelScale)) {
        config_.gumbelScale = 1.0;
    }
    if (config_.gumbelValueScale < 0.0 ||
        !std::isfinite(config_.gumbelValueScale)) {
        config_.gumbelValueScale = 0.1;
    }
    if (config_.gumbelMaxVisitInit < 0.0 ||
        !std::isfinite(config_.gumbelMaxVisitInit)) {
        config_.gumbelMaxVisitInit = 50.0;
    }
    if (config_.paperFullGumbel) {
        // Canonical Gumbel AlphaZero/MuZero settings.  These overrides make
        // the experiment reproducible even when a checkpoint carries the
        // project's exploratory extensions from an earlier run.
        config_.rootSearchMode = 2;
        // mctx.gumbel_muzero_policy defaults to considering at most 16 root
        // actions. Keep this canonical paper default separate from the
        // project's legacy `0 = all legal actions` convention.
        config_.gumbelCandidates = 16;
        config_.gumbelScale = 1.0;
        // mctx.qtransform_completed_by_mix_value defaults to value_scale=0.1.
        config_.gumbelValueScale = 0.1;
        config_.gumbelMaxVisitInit = 50.0;
        config_.gumbelRescaleQ = true;
        config_.gumbelUseMixedValue = true;
        config_.gumbelInteriorUniformity = 0.0;
        config_.gumbelInteriorMinVisits = 0;
        config_.gumbelInteriorCoverageFraction = 0.0;
        config_.gumbelInteriorValueMode = 0;
        config_.gumbelInteriorMaxMixVisits = 64.0;
        config_.dirichletAlpha = 0.0;
        config_.dirichletEpsilon = 0.0;
        config_.priorTemperature = 1.0;
    }
    root_.networkValue = std::isfinite(rootValue)
        ? std::clamp(rootValue, 0.0, 1.0)
        : hpValue(rootState_);
    root_.networkValueValid = true;
    root_.propagatedValue = root_.networkValue;
    root_.propagatedValid = true;
    expandWithLogits(root_, rootState_, legalActions(rootState_), rootPolicyLogits);
    applyRootDirichletNoise();
    initializeGumbelRoot();
}

void NoslNeuralSearchSession::initializeGumbelRoot() {
    gumbelActiveEdges_.clear();
    gumbelInitialEdges_.clear();
    gumbelNoise_.clear();
    gumbelScores_.clear();
    gumbelVisitSchedule_.clear();
    gumbelRound_ = 0;
    gumbelRoundQuota_ = 0;
    gumbelRoundStarted_ = 0;
    gumbelRoundCompleted_ = 0;
    gumbelRemaining_ = config_.simulations;
    if ((config_.rootSearchMode != 1 && config_.rootSearchMode != 2) ||
        root_.edges.empty() || config_.simulations < 1) {
        return;
    }

    // Gumbel is sampled once per root search from the search nonce.  It is
    // deliberately independent of the per-simulation NOSL redeterminization
    // stream, so it only controls root budget allocation.
    std::mt19937_64 rng(noslMix(searchNonce_ ^ 0x47554D42454C415AULL));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    gumbelNoise_.resize(root_.edges.size(), 0.0);
    gumbelScores_.resize(root_.edges.size(), 0.0);
    for (std::size_t i = 0; i < root_.edges.size(); ++i) {
        double u = uniform(rng);
        u = std::clamp(u, std::numeric_limits<double>::min(),
                       1.0 - std::numeric_limits<double>::epsilon());
        const double prior = std::max(root_.edges[i].prior, 1.0e-12);
        gumbelNoise_[i] = config_.gumbelScale * (-std::log(-std::log(u)));
        gumbelScores_[i] = std::log(prior) + gumbelNoise_[i];
    }
    std::vector<int> order(root_.edges.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        if (gumbelScores_[lhs] != gumbelScores_[rhs]) {
            return gumbelScores_[lhs] > gumbelScores_[rhs];
        }
        return root_.edges[lhs].action.bits < root_.edges[rhs].action.bits;
    });
    int candidateCount = static_cast<int>(order.size());
    if (config_.gumbelCandidates > 0) {
        candidateCount = std::min(candidateCount, config_.gumbelCandidates);
    }
    // mctx keeps max_num_considered_actions independent of the simulation
    // budget (for example K=16 with only 8 simulations).  The legacy
    // sequential-halving allocator needs the old clamp to avoid empty rounds;
    // strict paper mode follows mctx exactly.
    if (!config_.paperFullGumbel) {
        candidateCount = std::min(candidateCount, config_.simulations);
    }
    if (candidateCount < 1) {
        gumbelNoise_.clear();
        gumbelScores_.clear();
        return;
    }
    gumbelInitialEdges_.assign(order.begin(), order.begin() + candidateCount);
    gumbelActiveEdges_ = gumbelInitialEdges_;

    if (config_.rootSearchMode == 2) {
        initializeFullGumbelSchedule();
        return;
    }

    // Establish the first equal-budget round.  Later rounds recompute their
    // quota from the simulations left after the preceding elimination.
    advanceGumbelRound();
}

void NoslNeuralSearchSession::initializeFullGumbelSchedule() {
    gumbelVisitSchedule_.clear();
    const int maxConsidered = static_cast<int>(gumbelActiveEdges_.size());
    if (maxConsidered <= 0 || config_.simulations <= 0) {
        return;
    }
    if (maxConsidered == 1) {
        gumbelVisitSchedule_.reserve(config_.simulations);
        for (int visit = 0; visit < config_.simulations; ++visit) {
            gumbelVisitSchedule_.push_back(visit);
        }
        return;
    }

    // This is mctx.seq_halving.get_sequence_of_considered_visits().  The
    // resulting sequence is independent of the sampled Gumbel values; the
    // root selector only admits edges whose current visit count equals the
    // entry at the current simulation index.
    const int log2Max = static_cast<int>(std::ceil(
        std::log2(static_cast<double>(maxConsidered))));
    if (log2Max <= 0) {
        return;
    }
    std::vector<int> visits(static_cast<std::size_t>(maxConsidered), 0);
    int numConsidered = maxConsidered;
    gumbelVisitSchedule_.reserve(config_.simulations);
    while (static_cast<int>(gumbelVisitSchedule_.size()) < config_.simulations) {
        const long long denominator = static_cast<long long>(log2Max) *
            static_cast<long long>(numConsidered);
        const int numExtraVisits = std::max(
            1, static_cast<int>(static_cast<long long>(config_.simulations) /
                                std::max<long long>(1, denominator)));
        for (int extra = 0; extra < numExtraVisits &&
             static_cast<int>(gumbelVisitSchedule_.size()) < config_.simulations;
             ++extra) {
            for (int i = 0; i < numConsidered &&
                 static_cast<int>(gumbelVisitSchedule_.size()) < config_.simulations;
                 ++i) {
                gumbelVisitSchedule_.push_back(visits[static_cast<std::size_t>(i)]);
            }
            for (int i = 0; i < numConsidered; ++i) {
                ++visits[static_cast<std::size_t>(i)];
            }
        }
        numConsidered = std::max(2, numConsidered / 2);
    }
}

void NoslNeuralSearchSession::advanceGumbelRound() {
    if (gumbelActiveEdges_.empty()) {
        gumbelRoundQuota_ = 0;
        return;
    }
    if (gumbelRound_ > 0 && gumbelActiveEdges_.size() > 1) {
        std::sort(gumbelActiveEdges_.begin(), gumbelActiveEdges_.end(),
                  [&](int lhs, int rhs) {
            const auto &a = root_.edges[lhs];
            const auto &b = root_.edges[rhs];
            const double qa = edgeValueForSelection(a, root_);
            const double qb = edgeValueForSelection(b, root_);
            if (qa != qb) {
                return qa > qb;
            }
            if (gumbelScores_[lhs] != gumbelScores_[rhs]) {
                return gumbelScores_[lhs] > gumbelScores_[rhs];
            }
            return a.action.bits < b.action.bits;
        });
        const std::size_t keep = (gumbelActiveEdges_.size() + 1) / 2;
        gumbelActiveEdges_.resize(std::max<std::size_t>(1, keep));
    }
    gumbelRoundStarted_ = 0;
    gumbelRoundCompleted_ = 0;
    if (gumbelActiveEdges_.size() == 1) {
        gumbelRoundQuota_ = std::max(1, gumbelRemaining_);
        return;
    }
    int roundsLeft = 0;
    for (std::size_t count = gumbelActiveEdges_.size(); count > 1;
         count = (count + 1) / 2) {
        ++roundsLeft;
    }
    const int edgeCount = static_cast<int>(gumbelActiveEdges_.size());
    const int maxQuota = gumbelRemaining_ / edgeCount;
    gumbelRoundQuota_ = std::max(1, maxQuota / std::max(1, roundsLeft));
    gumbelRoundQuota_ = std::min(gumbelRoundQuota_, maxQuota);
}

int NoslNeuralSearchSession::selectRootEdge() {
    if (gumbelActiveEdges_.empty()) {
        return -1;
    }
    if (config_.rootSearchMode == 2) {
        if (gumbelVisitSchedule_.empty()) {
            initializeFullGumbelSchedule();
        }
        if (gumbelVisitSchedule_.empty()) {
            return gumbelActiveEdges_.front();
        }

        // `nextBatch()` installs a virtual visit on the root before entering
        // this function.  Use effective counts for scheduling so multiple
        // leaves requested in one IPC batch do not repeatedly select the same
        // root edge.  Once the batch is backed up, effective counts equal the
        // mctx visit counts exactly.
        const int simulationIndex = std::clamp(
            root_.visits + root_.virtualVisits - 1, 0,
            static_cast<int>(gumbelVisitSchedule_.size()) - 1);
        const int consideredVisit = gumbelVisitSchedule_[simulationIndex];
        const std::vector<double> scores = rootCompletedScores(true);
        int best = -1;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (const int index : gumbelActiveEdges_) {
            const auto &edge = root_.edges[static_cast<std::size_t>(index)];
            const int effectiveVisits = edge.visits + edge.virtualVisits;
            if (effectiveVisits != consideredVisit) {
                continue;
            }
            const double score = std::max(-1.0e9, scores[
                static_cast<std::size_t>(index)]);
            if (score > bestScore ||
                (score == bestScore &&
                 (best < 0 || edge.action.bits <
                              root_.edges[static_cast<std::size_t>(best)].action.bits))) {
                best = index;
                bestScore = score;
            }
        }
        if (best >= 0) {
            return best;
        }

        // A hidden-state re-determinization can invalidate an edge between
        // expansions, leaving no exact-count candidate.  Preserve progress by
        // choosing the least-visited active edge and then the score; this is
        // only a defensive path and does not affect ordinary mctx schedules.
        best = gumbelActiveEdges_.front();
        for (const int index : gumbelActiveEdges_) {
            const auto &candidate = root_.edges[static_cast<std::size_t>(index)];
            const auto &current = root_.edges[static_cast<std::size_t>(best)];
            const int candidateVisits = candidate.visits + candidate.virtualVisits;
            const int bestVisits = current.visits + current.virtualVisits;
            if (candidateVisits < bestVisits ||
                (candidateVisits == bestVisits &&
                 scores[static_cast<std::size_t>(index)] >
                     scores[static_cast<std::size_t>(best)])) {
                best = index;
            }
        }
        return best;
    }
    const int roundBudget = gumbelRoundQuota_ *
        static_cast<int>(gumbelActiveEdges_.size());
    if (gumbelRoundStarted_ >= roundBudget) {
        // nextBatch() never runs while pending leaves exist, so all virtual
        // visits from this round have been resolved before eliminating.
        if (gumbelRoundCompleted_ < gumbelRoundStarted_) {
            return -1;
        }
        gumbelRemaining_ = std::max(
            0, gumbelRemaining_ - gumbelRoundStarted_);
        ++gumbelRound_;
        advanceGumbelRound();
        if (gumbelRemaining_ <= 0) {
            return -1;
        }
    }
    int best = gumbelActiveEdges_.front();
    auto visits = [&](int index) {
        const auto &edge = root_.edges[index];
        return edge.visits + edge.virtualVisits;
    };
    for (const int index : gumbelActiveEdges_) {
        if (visits(index) < visits(best) ||
            (visits(index) == visits(best) &&
             gumbelScores_[index] > gumbelScores_[best])) {
            best = index;
        }
    }
    ++gumbelRoundStarted_;
    return best;
}

void NoslNeuralSearchSession::applyRootDirichletNoise() {
    if (root_.edges.empty() || config_.dirichletEpsilon <= 0.0 ||
        config_.dirichletAlpha <= 0.0) {
        return;
    }
    std::mt19937_64 rng(noslMix(searchNonce_ ^ 0xD1A1C137ULL));
    std::gamma_distribution<double> gamma(config_.dirichletAlpha, 1.0);
    std::vector<double> samples(root_.edges.size(), 0.0);
    double total = 0.0;
    for (double &sample : samples) {
        sample = gamma(rng);
        total += sample;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return;
    }
    const double epsilon = config_.dirichletEpsilon;
    for (std::size_t i = 0; i < root_.edges.size(); ++i) {
        root_.edges[i].prior = (1.0 - epsilon) * root_.edges[i].prior +
                               epsilon * samples[i] / total;
    }
}

double NoslNeuralSearchSession::hpValue(const BattleContext &bc) {
    if (bc.player.maxHp <= 0) {
        return 0.0;
    }
    return std::clamp(
        static_cast<double>(bc.player.curHp) / HP_VALUE_SCALE, 0.0, 1.0);
}

std::vector<Action> NoslNeuralSearchSession::legalActions(
    const BattleContext &bc) {
    return Action::enumerateCombatActions(bc);
}

int NoslNeuralSearchSession::thinActionId(const Action &action) {
    // Keep this mapping aligned with env/core/action_space.py.  Native search
    // actions are complete (card + target), while the whole-run network head
    // scores the autoregressive root slot; target variants share that score.
    constexpr int playBase = 1;
    constexpr int potionBase = 11;
    constexpr int pickBase = 21;
    constexpr int confirmSelect = 180;
    switch (action.getActionType()) {
        case ActionType::CARD:
            return playBase + action.getSourceIdx();
        case ActionType::POTION:
            return potionBase + action.getSourceIdx();
        case ActionType::SINGLE_CARD_SELECT:
            return pickBase + action.getSelectIdx();
        case ActionType::MULTI_CARD_SELECT:
            return confirmSelect;
        case ActionType::END_TURN:
            return 0;
        default:
            return -1;
    }
}

bool NoslNeuralSearchSession::actionTargetsEnemy(
    const BattleContext &state, const Action &action) {
    switch (action.getActionType()) {
        case ActionType::CARD:
            return action.getSourceIdx() >= 0 &&
                   action.getSourceIdx() < state.cards.cardsInHand &&
                   state.cards.hand[action.getSourceIdx()].requiresTarget();
        case ActionType::POTION:
            return action.getSourceIdx() >= 0 &&
                   action.getSourceIdx() < state.potionCapacity &&
                   potionRequiresTarget(state.potions[action.getSourceIdx()]);
        default:
            return false;
    }
}

std::string NoslNeuralSearchSession::actionLabel(
    const BattleContext &state, const Action &action) {
    std::ostringstream output;
    action.printDesc(output, state);
    return output.str();
}

void NoslNeuralSearchSession::expandUniform(
    Node &node, const BattleContext &state,
    const std::vector<Action> &actions) {
    node.edges.clear();
    node.edges.reserve(actions.size());
    const double prior = actions.empty() ? 0.0 : 1.0 / actions.size();
    for (const auto &action : actions) {
        node.edges.emplace_back(
            action, prior, actionTargetsEnemy(state, action),
            actionLabel(state, action));
        node.edges.back().semanticKey = publicActionKey(state, action);
    }
}

void NoslNeuralSearchSession::expandWithLogits(
    Node &node, const BattleContext &state,
    const std::vector<Action> &actions,
    const std::vector<double> &logits) {
    expandWithLogits(node, state, actions, logits.data(), logits.size());
}

void NoslNeuralSearchSession::expandWithLogits(
    Node &node, const BattleContext &state,
    const std::vector<Action> &actions,
    const double *logits, std::size_t logitsSize) {
    if (actions.empty()) {
        node.edges.clear();
        return;
    }
    if (logits == nullptr || logitsSize == 0) {
        expandUniform(node, state, actions);
        return;
    }

    // The thin action codec currently has 181 ids. A fixed table avoids a
    // hash allocation for every expanded leaf; invalid/unmapped actions keep
    // the old count=1 behavior.
    constexpr int kThinActionCount = 181;
    std::array<int, kThinActionCount> groupCounts{};
    double maxLogit = -std::numeric_limits<double>::infinity();
    for (const auto &action : actions) {
        const int id = thinActionId(action);
        if (id < 0) {
            continue;
        }
        if (id >= 0 && id < kThinActionCount) {
            ++groupCounts[static_cast<std::size_t>(id)];
        }
        if (id < static_cast<int>(logitsSize) &&
            std::isfinite(logits[id])) {
            maxLogit = std::max(
                maxLogit, logits[id] / config_.priorTemperature);
        }
    }
    if (!std::isfinite(maxLogit)) {
        expandUniform(node, state, actions);
        return;
    }

    node.edges.clear();
    node.edges.reserve(actions.size());
    double total = 0.0;
    std::vector<double> weights;
    weights.reserve(actions.size());
    for (const auto &action : actions) {
        const int id = thinActionId(action);
        const int count = id >= 0 && id < kThinActionCount &&
                groupCounts[static_cast<std::size_t>(id)] > 0
            ? groupCounts[static_cast<std::size_t>(id)] : 1;
        const double logit = id >= 0 && id < static_cast<int>(logitsSize) &&
                std::isfinite(logits[id])
            ? logits[id] / config_.priorTemperature
            : maxLogit;
        const double weight = std::exp(std::clamp(logit - maxLogit, -80.0, 80.0)) /
                              static_cast<double>(count);
        weights.push_back(weight);
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        expandUniform(node, state, actions);
        return;
    }
    for (std::size_t i = 0; i < actions.size(); ++i) {
        node.edges.emplace_back(
            actions[i], weights[i] / total,
            actionTargetsEnemy(state, actions[i]),
            actionLabel(state, actions[i]));
        node.edges.back().semanticKey = publicActionKey(state, actions[i]);
    }
}

void NoslNeuralSearchSession::recomputeEdgeValue(Edge &edge) const {
    // An action can lead to several public states because each NOSL
    // simulation redeterminizes hidden RNG independently.  Those children
    // form a chance node: combine their propagated values by the observed
    // world-line frequency, never by max.
    if (!edge.children.empty()) {
        double weightedValue = 0.0;
        double totalWeight = 0.0;
        for (const auto &[key, child] : edge.children) {
            (void)key;
            if (!child) {
                continue;
            }
            if (!child->propagatedValid && child->valueVisits <= 0 &&
                !child->networkValueValid) {
                continue;
            }
            const double value = nodeValueForSelection(*child);
            if (!std::isfinite(value)) {
                continue;
            }
            // valueVisits excludes forced coverage probes.  If a child has
            // only such probes, its visit count still gives the best observed
            // estimate of the chance probability.
            double weight = static_cast<double>(child->valueVisits > 0
                ? child->valueVisits : child->visits);
            if (!(weight > 0.0) || !std::isfinite(weight)) {
                weight = 1.0;
            }
            weightedValue += weight * value;
            totalWeight += weight;
        }
        if (totalWeight > 0.0 && std::isfinite(totalWeight)) {
            edge.propagatedValue = std::clamp(
                weightedValue / totalWeight, 0.0, 1.0);
            edge.propagatedValid = true;
            return;
        }
    }

    // End-turn leaves and terminal actions do not have a child decision node;
    // their direct leaf samples are already the complete action value.
    if (edge.valueVisits > 0) {
        edge.propagatedValue = std::clamp(
            edge.valueSum / static_cast<double>(edge.valueVisits), 0.0, 1.0);
        edge.propagatedValid = std::isfinite(edge.propagatedValue);
        return;
    }
    edge.propagatedValue = 0.0;
    edge.propagatedValid = false;
}

void NoslNeuralSearchSession::recomputeNodeValue(Node &node) const {
    if (config_.paperFullGumbel) {
        // mctx backs a decision node up as the visit-weighted mean of its
        // outgoing action values.  The experimental path below intentionally
        // keeps its historical equal-action/progressive aggregation.
        double weightedValue = 0.0;
        double totalVisits = 0.0;
        for (const auto &edge : node.edges) {
            if (edge.visits <= 0) {
                continue;
            }
            const double value = edgeValueForSelection(edge, node);
            if (!std::isfinite(value)) {
                continue;
            }
            weightedValue += static_cast<double>(edge.visits) * value;
            totalVisits += static_cast<double>(edge.visits);
        }
        if (totalVisits > 0.0 && std::isfinite(totalVisits)) {
            node.propagatedValue = std::clamp(
                weightedValue / totalVisits, 0.0, 1.0);
            node.propagatedValid = std::isfinite(node.propagatedValue);
            return;
        }
        node.propagatedValue = node.networkValueValid
            ? std::clamp(node.networkValue, 0.0, 1.0) : 0.0;
        node.propagatedValid = node.networkValueValid &&
            std::isfinite(node.propagatedValue);
        return;
    }

    std::vector<double> values;
    values.reserve(node.edges.size());
    for (const auto &edge : node.edges) {
        if (edge.propagatedValid && std::isfinite(edge.propagatedValue)) {
            values.push_back(edge.propagatedValue);
        } else if (edge.valueVisits > 0) {
            const double sampleValue = edge.valueSum /
                static_cast<double>(edge.valueVisits);
            if (std::isfinite(sampleValue)) {
                values.push_back(std::clamp(sampleValue, 0.0, 1.0));
            }
        }
    }

    if (values.empty()) {
        if (node.valueVisits > 0) {
            node.propagatedValue = std::clamp(
                node.valueSum / static_cast<double>(node.valueVisits),
                0.0, 1.0);
            node.propagatedValid = std::isfinite(node.propagatedValue);
        } else if (node.networkValueValid &&
                   std::isfinite(node.networkValue)) {
            node.propagatedValue = std::clamp(node.networkValue, 0.0, 1.0);
            node.propagatedValid = true;
        } else {
            node.propagatedValue = 0.0;
            node.propagatedValid = false;
        }
        return;
    }

    const double meanValue = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    const double bestValue = *std::max_element(values.begin(), values.end());
    double alpha = 0.0;
    if (config_.gumbelInteriorValueMode == 1) {
        alpha = 1.0;
    } else if (config_.gumbelInteriorValueMode == 2) {
        alpha = std::clamp(
            static_cast<double>(node.visits) /
                std::max(1.0, config_.gumbelInteriorMaxMixVisits),
            0.0, 1.0);
    }
    node.propagatedValue = std::clamp(
        (1.0 - alpha) * meanValue + alpha * bestValue, 0.0, 1.0);
    node.propagatedValid = std::isfinite(node.propagatedValue);
}

double NoslNeuralSearchSession::edgeValueForSelection(
    const Edge &edge, const Node &parent) const {
    if (edge.propagatedValid && std::isfinite(edge.propagatedValue)) {
        return edge.propagatedValue;
    }
    if (edge.valueVisits > 0) {
        const double sampleValue = edge.valueSum /
            static_cast<double>(edge.valueVisits);
        if (std::isfinite(sampleValue)) {
            return std::clamp(sampleValue, 0.0, 1.0);
        }
    }
    // First-play urgency remains anchored to the parent's network estimate.
    // Do not let a high-valued sibling selected by max aggregation leak into
    // every still-unvisited action before that action receives a value probe.
    const double fallback = (parent.networkValueValid
        ? parent.networkValue : nodeValueForSelection(parent)) -
        config_.fpuReduction;
    return std::isfinite(fallback) ? std::clamp(fallback, 0.0, 1.0) : 0.0;
}

double NoslNeuralSearchSession::nodeValueForSelection(const Node &node) const {
    if (node.propagatedValid && std::isfinite(node.propagatedValue)) {
        return node.propagatedValue;
    }
    if (node.valueVisits > 0) {
        const double sampleValue = node.valueSum /
            static_cast<double>(node.valueVisits);
        if (std::isfinite(sampleValue)) {
            return std::clamp(sampleValue, 0.0, 1.0);
        }
    }
    if (node.networkValueValid && std::isfinite(node.networkValue)) {
        return std::clamp(node.networkValue, 0.0, 1.0);
    }
    return 0.0;
}

int NoslNeuralSearchSession::selectPuct(const Node &node) const {
    if (node.edges.empty()) {
        return -1;
    }
    const double parentVisits = std::sqrt(static_cast<double>(
        std::max(1, node.visits + node.virtualVisits)));
    int best = -1;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(node.edges.size()); ++i) {
        const auto &edge = node.edges[i];
        const int effectiveVisits = edge.visits + edge.virtualVisits;
        const double q = edgeValueForSelection(edge, node);
        const double score = q + config_.cPuct * edge.prior * parentVisits /
            static_cast<double>(1 + effectiveVisits);
        if (score > bestScore ||
            (score == bestScore && (best < 0 || edge.action.bits <
                                    node.edges[best].action.bits))) {
            best = i;
            bestScore = score;
        }
    }
    return best;
}

NoslNeuralSearchSession::QTransformDetails
NoslNeuralSearchSession::completedQDetails(const Node &node) const {
    const std::size_t count = node.edges.size();
    QTransformDetails details;
    details.raw.assign(count, node.networkValue);
    details.normalized.assign(count, 0.0);
    details.scaled.assign(count, 0.0);
    if (count == 0) {
        return details;
    }

    // Use the structurally propagated action values.  An action with several
    // NOSL public-state children has already been chance-averaged by
    // recomputeEdgeValue(); this level is the active decision node.
    const bool paper = config_.paperFullGumbel;
    double sumVisits = 0.0;
    double sumPrior = 0.0;
    double weightedQ = 0.0;
    for (const auto &edge : node.edges) {
        const int observedVisits = paper ? edge.visits : edge.valueVisits;
        if (observedVisits <= 0) {
            continue;
        }
        const double q = edgeValueForSelection(edge, node);
        const double prior = std::max(edge.prior, 1.0e-12);
        sumVisits += static_cast<double>(observedVisits);
        sumPrior += prior;
        weightedQ += prior * (std::isfinite(q)
            ? q : nodeValueForSelection(node));
    }
    // The paper's mixed value is bootstrapped from the network value at this
    // node.  The non-paper path may instead use structural mean/max
    // propagation, which is intentionally kept for existing experiments.
    double mixedValue = paper ? node.networkValue : nodeValueForSelection(node);
    if (config_.gumbelUseMixedValue && sumVisits > 0.0) {
        const double normalizedWeightedQ = sumPrior > 0.0
            ? weightedQ / sumPrior
            : (paper ? node.networkValue : nodeValueForSelection(node));
        const double bootstrap = paper
            ? node.networkValue : nodeValueForSelection(node);
        mixedValue = (bootstrap +
                      sumVisits * normalizedWeightedQ) /
            (sumVisits + 1.0);
    }
    if (!std::isfinite(mixedValue)) {
        mixedValue = paper ? node.networkValue : nodeValueForSelection(node);
    }

    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        const auto &edge = node.edges[i];
        double value = edge.propagatedValid ||
                (paper ? edge.visits > 0 : edge.valueVisits > 0)
            ? edgeValueForSelection(edge, node)
            : (config_.gumbelUseMixedValue
                ? mixedValue : nodeValueForSelection(node));
        if (!std::isfinite(value)) {
            value = mixedValue;
        }
        details.raw[i] = value;
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    const double denominator = std::max(maxValue - minValue, 1.0e-8);
    for (std::size_t i = 0; i < count; ++i) {
        // Keep the normalized view even when rescale_values is disabled.  It
        // makes diagnostics comparable across ablations; `scaled` below is
        // the exact quantity used by action selection.
        details.normalized[i] = (details.raw[i] - minValue) / denominator;
    }

    int maxVisit = 0;
    for (const auto &edge : node.edges) {
        maxVisit = std::max(maxVisit, edge.visits);
    }
    const double scale = (config_.gumbelMaxVisitInit +
                          static_cast<double>(maxVisit)) *
        config_.gumbelValueScale;
    for (std::size_t i = 0; i < count; ++i) {
        const double base = config_.gumbelRescaleQ
            ? details.normalized[i] : details.raw[i];
        details.scaled[i] = base * scale;
    }
    return details;
}

std::vector<double> NoslNeuralSearchSession::completedQTransform(
    const Node &node) const {
    return completedQDetails(node).scaled;
}

std::vector<double> NoslNeuralSearchSession::rootCompletedScores(
    bool includeNoise) const {
    const std::vector<double> q = completedQTransform(root_);
    std::vector<double> scores(root_.edges.size(),
                               -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < root_.edges.size(); ++i) {
        const double prior = std::max(root_.edges[i].prior, 1.0e-12);
        const double noise = includeNoise && i < gumbelNoise_.size()
            ? gumbelNoise_[i]
            : 0.0;
        scores[i] = std::log(prior) + noise + q[i];
    }
    return scores;
}

std::vector<double> NoslNeuralSearchSession::rootPolicyTarget() const {
    const std::vector<double> scores = rootCompletedScores(false);
    std::vector<double> target(scores.size(), 0.0);
    if (scores.empty()) {
        return target;
    }
    double maxScore = -std::numeric_limits<double>::infinity();
    for (double score : scores) {
        if (std::isfinite(score)) {
            maxScore = std::max(maxScore, score);
        }
    }
    if (!std::isfinite(maxScore)) {
        const double uniform = 1.0 / static_cast<double>(scores.size());
        std::fill(target.begin(), target.end(), uniform);
        return target;
    }
    double total = 0.0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) {
            continue;
        }
        target[i] = std::exp(std::clamp(scores[i] - maxScore, -80.0, 0.0));
        total += target[i];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        const double uniform = 1.0 / static_cast<double>(scores.size());
        std::fill(target.begin(), target.end(), uniform);
        return target;
    }
    for (double &value : target) {
        value /= total;
    }
    return target;
}

int NoslNeuralSearchSession::selectFullGumbelInterior(
    const Node &node, bool *usedCoverage) const {
    if (usedCoverage != nullptr) {
        *usedCoverage = false;
    }
    if (node.edges.empty()) {
        return -1;
    }
    // Prior-independent coverage phase.  This is deliberately performed
    // before constructing completed-policy scores: a zero/near-zero prior
    // must not be able to suppress the first real value evaluation forever.
    const int minVisits = std::max(0, config_.gumbelInteriorMinVisits);
    const double coverageFraction = std::clamp(
        config_.gumbelInteriorCoverageFraction, 0.0, 1.0);
    if (minVisits > 0 || coverageFraction > 0.0) {
        int totalVisits = 0;
        for (const auto &edge : node.edges) {
            totalVisits += edge.visits + edge.virtualVisits;
        }
        const int edgeCount = static_cast<int>(node.edges.size());
        const int dynamicTarget = coverageFraction > 0.0
            ? static_cast<int>(std::ceil(
                coverageFraction * static_cast<double>(totalVisits + 1) /
                static_cast<double>(edgeCount)))
            : 0;
        const int targetVisits = std::max(minVisits, dynamicTarget);
        int bestUnvisited = -1;
        for (int i = 0; i < static_cast<int>(node.edges.size()); ++i) {
            const auto &edge = node.edges[static_cast<std::size_t>(i)];
            const int visits = edge.visits + edge.virtualVisits;
            if (visits >= targetVisits) {
                continue;
            }
            if (bestUnvisited < 0 ||
                visits < node.edges[static_cast<std::size_t>(bestUnvisited)]
                              .visits +
                    node.edges[static_cast<std::size_t>(bestUnvisited)]
                              .virtualVisits ||
                (visits ==
                     node.edges[static_cast<std::size_t>(bestUnvisited)]
                             .visits +
                         node.edges[static_cast<std::size_t>(bestUnvisited)]
                             .virtualVisits &&
                 edge.action.bits <
                     node.edges[static_cast<std::size_t>(bestUnvisited)]
                         .action.bits)) {
                bestUnvisited = i;
            }
        }
        if (bestUnvisited >= 0) {
            if (usedCoverage != nullptr) {
                *usedCoverage = true;
            }
            return bestUnvisited;
        }
    }
    const std::vector<double> q = completedQTransform(node);
    std::vector<double> logits(node.edges.size(), 0.0);
    double maxLogit = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        logits[i] = std::log(std::max(node.edges[i].prior, 1.0e-12)) + q[i];
        maxLogit = std::max(maxLogit, logits[i]);
    }
    double probabilityTotal = 0.0;
    for (double &logit : logits) {
        logit = std::exp(std::clamp(logit - maxLogit, -80.0, 0.0));
        probabilityTotal += logit;
    }
    if (!(probabilityTotal > 0.0) || !std::isfinite(probabilityTotal)) {
        probabilityTotal = static_cast<double>(node.edges.size());
        std::fill(logits.begin(), logits.end(), 1.0);
    }

    int totalVisits = 0;
    for (const auto &edge : node.edges) {
        totalVisits += edge.visits + edge.virtualVisits;
    }
    int best = -1;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        const auto &edge = node.edges[i];
        const int visits = edge.visits + edge.virtualVisits;
        double probability = logits[i] / probabilityTotal;
        // Full-Gumbel's interior selector normally follows
        // `completed_policy - visit_fraction`.  With sparse, high-entropy
        // combat states this can let one edge monopolize a node before other
        // edges receive a model value.  Optionally anneal the target toward a
        // uniform allocation as the node accumulates visits.  The default
        // (uniformity=0) is exactly the original rule.
        const double uniformity = std::clamp(
            config_.gumbelInteriorUniformity, 0.0, 1.0);
        if (uniformity > 0.0 && node.edges.size() > 1) {
            const double visitScale = std::max(
                1.0, config_.gumbelMaxVisitInit);
            const double progress = std::clamp(
                static_cast<double>(totalVisits) / visitScale, 0.0, 1.0);
            const double mix = uniformity * progress;
            probability = (1.0 - mix) * probability +
                mix / static_cast<double>(node.edges.size());
        }
        const double score = probability - static_cast<double>(visits) /
            (1.0 + static_cast<double>(totalVisits));
        if (score > bestScore ||
            (score == bestScore &&
             (best < 0 || edge.action.bits < node.edges[best].action.bits))) {
            best = static_cast<int>(i);
            bestScore = score;
        }
    }
    return best;
}

void NoslNeuralSearchSession::backupStats(Pending &pending, double value) {
    if (!std::isfinite(value)) {
        value = 0.0;
    }
    value = std::clamp(value, 0.0, 1.0);
    if (pending.leaf != nullptr && pending.needsPolicy &&
        pending.leaf->edges.empty()) {
        // The actual policy is installed by backprop(), which calls this
        // helper only after expansion.  Keeping the flag clear here prevents
        // duplicate expansion requests once the leaf is resolved.
        pending.leaf->expansionPending = false;
    }
    for (auto *node : pending.nodes) {
        node->virtualVisits = std::max(0, node->virtualVisits - 1);
        ++node->visits;
        if (!pending.excludeValue) {
            ++node->valueVisits;
            node->valueSum += value;
        }
    }
    for (auto *edge : pending.edges) {
        edge->virtualVisits = std::max(0, edge->virtualVisits - 1);
        ++edge->visits;
        if (!pending.excludeValue) {
            ++edge->valueVisits;
            edge->valueSum += value;
        }
    }
    if (config_.rootSearchMode == 1 && !gumbelActiveEdges_.empty()) {
        ++gumbelRoundCompleted_;
    }
    ++simulationsCompleted_;
}

void NoslNeuralSearchSession::recomputePathValues(
    const Pending &pending) {
    // Rebuild values from the bottom of this sampled path.  The raw sample
    // sums above remain available for diagnostics, while propagated values
    // now respect the two kinds of backward edge:
    //   action -> children: chance average over NOSL world lines;
    //   decision node -> actions: configured mean/max/progressive aggregate.
    for (int i = static_cast<int>(pending.edges.size()) - 1; i >= 0; --i) {
        const std::size_t edgeIndex = static_cast<std::size_t>(i);
        if (edgeIndex + 1 < pending.nodes.size()) {
            recomputeNodeValue(*pending.nodes[edgeIndex + 1]);
        }
        recomputeEdgeValue(*pending.edges[edgeIndex]);
        if (edgeIndex < pending.nodes.size()) {
            recomputeNodeValue(*pending.nodes[edgeIndex]);
        }
    }
    if (pending.edges.empty()) {
        for (auto itNode = pending.nodes.rbegin();
             itNode != pending.nodes.rend(); ++itNode) {
            recomputeNodeValue(**itNode);
        }
    }
}

void NoslNeuralSearchSession::backup(Pending &pending, double value) {
    backupStats(pending, value);
    recomputePathValues(pending);
}

void NoslNeuralSearchSession::finishTerminal(Pending &pending, double value) {
    backup(pending, value);
}

std::vector<NoslNeuralSearchSession::LeafRequest>
NoslNeuralSearchSession::nextBatch(int maxLeaves) {
    if (maxLeaves < 1) {
        maxLeaves = 1;
    }
    if (!pending_.empty() || done()) {
        return {};
    }
    std::vector<LeafRequest> requests;
    requests.reserve(maxLeaves);
    // A batch owns at most maxLeaves pending paths. Keep the hash table at a
    // stable size across the hot backprop/nextBatch cycle and avoid a rehash
    // when larger evaluation batches are used.
    if (pending_.bucket_count() < static_cast<std::size_t>(maxLeaves)) {
        pending_.reserve(static_cast<std::size_t>(maxLeaves));
    }
    while (simulationsStarted_ < config_.simulations &&
           static_cast<int>(requests.size()) < maxLeaves) {
        BattleContext state(rootState_);
        noslRedeterminizeCanonical(
            state, noslMix(searchNonce_ + static_cast<std::uint64_t>(
                                   simulationsStarted_)));
        ++simulationsStarted_;

        Pending pending;
        pending.id = nextRequestId_++;
        const std::size_t pathReserve = config_.maxDepth > 0
            ? static_cast<std::size_t>(config_.maxDepth) + 1 : 16;
        pending.nodes.reserve(pathReserve);
        pending.edges.reserve(pathReserve);
        pending.nodes.push_back(&root_);
        root_.virtualVisits++;
        Node *node = &root_;
        int depth = 0;
        bool haveLeaf = false;
        bool schedulerBlocked = false;
        while (state.outcome == Outcome::UNDECIDED) {
            // A depth limit truncates the search, it does not turn the
            // current state into a terminal state.  Ask the value network for
            // this state and do not request policy logits because no child
            // expansion should be installed at a truncated leaf.
            if (config_.maxDepth > 0 && depth >= config_.maxDepth) {
                pending.state = state;
                pending.leaf = nullptr;
                pending.legalActions.clear();
                pending.needsPolicy = false;
                haveLeaf = true;
                break;
            }
            if (node->edges.empty()) {
                pending.state = state;
                pending.leaf = node;
                pending.legalActions = legalActions(state);
                pending.needsPolicy = true;
                haveLeaf = true;
                break;
            }
            int edgeIndex = -1;
            if (node == &root_ &&
                (config_.rootSearchMode == 1 ||
                 config_.rootSearchMode == 2) &&
                !gumbelActiveEdges_.empty()) {
                edgeIndex = selectRootEdge();
            } else if (config_.rootSearchMode == 2 && node != &root_) {
                bool usedCoverage = false;
                edgeIndex = selectFullGumbelInterior(*node, &usedCoverage);
                pending.excludeValue = pending.excludeValue || usedCoverage;
            } else {
                edgeIndex = selectPuct(*node);
            }
            if (edgeIndex < 0) {
                schedulerBlocked = node == &root_ &&
                    (config_.rootSearchMode == 1 ||
                     config_.rootSearchMode == 2) &&
                    !gumbelActiveEdges_.empty();
                break;
            }
            Edge *edge = edgeIndex >= 0 ? &node->edges[edgeIndex] : nullptr;
            Action executableAction;
            bool foundAction = false;
            if (edge != nullptr && edge->action.isValidAction(state) &&
                publicActionKey(state, edge->action) == edge->semanticKey) {
                executableAction = edge->action;
                foundAction = true;
            } else if (edge != nullptr) {
                const auto currentActions = legalActions(state);
                for (const auto &candidateAction : currentActions) {
                    const auto candidateKey = publicActionKey(
                        state, candidateAction);
                    if (candidateKey != edge->semanticKey) {
                        continue;
                    }
                    for (auto &candidateEdge : node->edges) {
                        if (&candidateEdge != edge &&
                            candidateEdge.semanticKey == candidateKey) {
                            edge = &candidateEdge;
                            break;
                        }
                    }
                    executableAction = candidateAction;
                    foundAction = true;
                    break;
                }
            }
            if (!foundAction) {
                edge = nullptr;
                for (auto &candidate : node->edges) {
                    if (candidate.action.isValidAction(state)) {
                        edge = &candidate;
                        executableAction = candidate.action;
                        foundAction = true;
                        break;
                    }
                }
            }
            if (!foundAction) {
                break;
            }
            pending.edges.push_back(edge);
            edge->virtualVisits++;
            state.stopAfterEnemyTurn =
                config_.endTurnAsLeaf &&
                executableAction.getActionType() == ActionType::END_TURN;
            executableAction.execute(state);
            ++depth;

            if (config_.endTurnAsLeaf &&
                executableAction.getActionType() == ActionType::END_TURN) {
                // Do not expand the hidden/random continuation of End Turn.
                // The resulting state is evaluated by the caller's value
                // network and backed up as a value-only leaf.
                pending.state = state;
                pending.leaf = nullptr;
                pending.legalActions.clear();
                pending.needsPolicy = false;
                haveLeaf = true;
                break;
            }

            const auto childKey = publicStateKey(state);
            auto childIt = edge->children.find(childKey);
            if (childIt == edge->children.end()) {
                childIt = edge->children.emplace(
                    childKey, std::make_unique<Node>()).first;
            }
            node = childIt->second.get();
            node->virtualVisits++;
            pending.nodes.push_back(node);
        }

        if (!haveLeaf) {
            if (schedulerBlocked) {
                // The current round has outstanding leaf requests.  Leave
                // this simulation unstarted; the caller will backpropagate
                // the returned batch and invoke nextBatch() again, at which
                // point the round can be advanced safely.
                root_.virtualVisits = std::max(0, root_.virtualVisits - 1);
                --simulationsStarted_;
                break;
            }
            finishTerminal(pending, hpValue(state));
            continue;
        }
        if (state.outcome != Outcome::UNDECIDED ||
            (pending.needsPolicy && pending.legalActions.empty())) {
            finishTerminal(pending, hpValue(state));
            continue;
        }

        const auto requestId = pending.id;
        pending_[requestId] = std::move(pending);
        const auto &stored = pending_.at(requestId);
        LeafRequest request;
        request.requestId = stored.id;
        request.state = &stored.state;
        request.legalActions = stored.legalActions;
        request.needsPolicy = stored.needsPolicy;
        requests.push_back(std::move(request));
    }
    return requests;
}

void NoslNeuralSearchSession::backprop(
    std::uint64_t requestId, double value,
    const std::vector<double> &policyLogits) {
    auto it = pending_.find(requestId);
    if (it == pending_.end()) {
        throw std::invalid_argument("unknown NOSL neural leaf request id");
    }
    Pending pending = std::move(it->second);
    pending_.erase(it);
    if (pending.leaf != nullptr && pending.needsPolicy &&
        pending.leaf->edges.empty()) {
        pending.leaf->networkValue = std::isfinite(value)
            ? std::clamp(value, 0.0, 1.0)
            : 0.0;
        pending.leaf->networkValueValid = true;
        expandWithLogits(
            *pending.leaf, pending.state, pending.legalActions, policyLogits);
    }
    backup(pending, value);
}

void NoslNeuralSearchSession::backpropBatch(
    const std::vector<std::uint64_t> &requestIds,
    const std::vector<double> &values,
    const std::vector<std::vector<double>> &policyLogits) {
    if (requestIds.size() != values.size() ||
        requestIds.size() != policyLogits.size()) {
        throw std::invalid_argument("NOSL backprop batch lengths differ");
    }
    std::vector<Pending> batch;
    batch.reserve(requestIds.size());
    for (std::size_t i = 0; i < requestIds.size(); ++i) {
        auto it = pending_.find(requestIds[i]);
        if (it == pending_.end()) {
            throw std::invalid_argument("unknown NOSL neural leaf request id");
        }
        Pending pending = std::move(it->second);
        pending_.erase(it);
        if (pending.leaf != nullptr && pending.needsPolicy &&
            pending.leaf->edges.empty()) {
            pending.leaf->networkValue = std::isfinite(values[i])
                ? std::clamp(values[i], 0.0, 1.0)
                : 0.0;
            pending.leaf->networkValueValid = true;
            expandWithLogits(
                *pending.leaf, pending.state, pending.legalActions,
                policyLogits[i]);
        }
        backupStats(pending, values[i]);
        batch.emplace_back(std::move(pending));
    }
    for (const auto &pending : batch) {
        recomputePathValues(pending);
    }
}

void NoslNeuralSearchSession::backpropBatchArray(
    const std::uint64_t *requestIds,
    const double *values,
    const double *policyLogits,
    std::size_t count,
    std::size_t policyWidth) {
    if (count > 0 && (requestIds == nullptr || values == nullptr ||
                      (policyWidth > 0 && policyLogits == nullptr))) {
        throw std::invalid_argument("null NOSL backprop array storage");
    }
    std::vector<Pending> batch;
    batch.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto it = pending_.find(requestIds[i]);
        if (it == pending_.end()) {
            throw std::invalid_argument("unknown NOSL neural leaf request id");
        }
        Pending pending = std::move(it->second);
        pending_.erase(it);
        if (pending.leaf != nullptr && pending.needsPolicy &&
            pending.leaf->edges.empty()) {
            pending.leaf->networkValue = std::isfinite(values[i])
                ? std::clamp(values[i], 0.0, 1.0)
                : 0.0;
            pending.leaf->networkValueValid = true;
            const double *row = policyWidth > 0
                ? policyLogits + i * policyWidth : nullptr;
            expandWithLogits(
                *pending.leaf, pending.state, pending.legalActions,
                row, policyWidth);
        }
        backupStats(pending, values[i]);
        batch.emplace_back(std::move(pending));
    }
    for (const auto &pending : batch) {
        recomputePathValues(pending);
    }
}

NoslCombatDecision NoslNeuralSearchSession::result() const {
    NoslCombatDecision result;
    result.totalRootVisits = root_.visits;
    result.rootValue = nodeValueForSelection(root_);
    if (root_.edges.empty()) {
        result.action = Action();
        return result;
    }
    int best = 0;
    if (config_.rootSearchMode == 2 && !gumbelActiveEdges_.empty()) {
        // Full Gumbel MuZero acts from the most-visited candidates, breaking
        // ties with the final `gumbel + log-prior + completed-Q` score.
        int maxVisit = -1;
        for (const int index : gumbelActiveEdges_) {
            maxVisit = std::max(maxVisit,
                                root_.edges[static_cast<std::size_t>(index)].visits);
        }
        const std::vector<double> scores = rootCompletedScores(true);
        best = gumbelActiveEdges_.front();
        for (const int index : gumbelActiveEdges_) {
            const auto &candidate = root_.edges[static_cast<std::size_t>(index)];
            const auto &current = root_.edges[static_cast<std::size_t>(best)];
            const bool candidateEligible = candidate.visits == maxVisit;
            const bool bestEligible = current.visits == maxVisit;
            if ((candidateEligible && !bestEligible) ||
                (candidateEligible == bestEligible &&
                 (scores[static_cast<std::size_t>(index)] >
                      scores[static_cast<std::size_t>(best)] ||
                  (scores[static_cast<std::size_t>(index)] ==
                       scores[static_cast<std::size_t>(best)] &&
                   candidate.action.bits < current.action.bits)))) {
                best = index;
            }
        }
        const std::vector<double> target = rootPolicyTarget();
        result.rootPolicy.reserve(root_.edges.size());
        for (std::size_t i = 0; i < root_.edges.size(); ++i) {
            result.rootPolicy.emplace_back(root_.edges[i].action.bits, target[i]);
        }
    } else {
        for (int i = 1; i < static_cast<int>(root_.edges.size()); ++i) {
            if (root_.edges[i].visits > root_.edges[best].visits ||
                (root_.edges[i].visits == root_.edges[best].visits &&
                 (root_.edges[i].valueSum > root_.edges[best].valueSum ||
                  (root_.edges[i].valueSum == root_.edges[best].valueSum &&
                   config_.rootSearchMode == 1 &&
                   !gumbelScores_.empty() &&
                   gumbelScores_[i] > gumbelScores_[best])))) {
                best = i;
            }
        }
    }
    result.action = root_.edges[best].action;
    if (root_.edges[best].targeted) {
        result.selectedActionTargeted = true;
        result.selectedTarget = result.action.getTargetIdx();
    }
    result.rootVisits.reserve(root_.edges.size());
    for (const auto &edge : root_.edges) {
        result.rootVisits.emplace_back(edge.action.bits, edge.visits);
    }
    return result;
}

NoslNeuralSearchSession::TreeSnapshot
NoslNeuralSearchSession::treeSnapshot(const std::vector<int> &path,
                                      int maxChildren) const {
    TreeSnapshot snapshot;
    const Node *node = &root_;
    const auto representativeChild = [](const Edge &edge) -> const Node * {
        const Node *best = nullptr;
        std::uint64_t bestKey = 0;
        for (const auto &[key, child] : edge.children) {
            if (!child || !best || child->visits > best->visits ||
                (child->visits == best->visits && key < bestKey)) {
                best = child.get();
                bestKey = key;
            }
        }
        return best;
    };
    for (const int edgeIndex : path) {
        if (edgeIndex < 0 || edgeIndex >= static_cast<int>(node->edges.size())) {
            return snapshot;
        }
        const auto &edge = node->edges[edgeIndex];
        const auto *child = representativeChild(edge);
        if (!child) {
            return snapshot;
        }
        node = child;
    }

    snapshot.valid = true;
    snapshot.visits = node->visits;
    snapshot.valueVisits = node->valueVisits;
    snapshot.value = nodeValueForSelection(*node);
    snapshot.sampleValue = node->valueVisits > 0
        ? node->valueSum / static_cast<double>(node->valueVisits)
        : nodeValueForSelection(*node);
    const QTransformDetails qDetails = completedQDetails(*node);

    std::vector<int> order;
    order.reserve(node->edges.size());
    for (int i = 0; i < static_cast<int>(node->edges.size()); ++i) {
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const auto &a = node->edges[lhs];
        const auto &b = node->edges[rhs];
        if (a.visits != b.visits) {
            return a.visits > b.visits;
        }
        if (a.valueSum != b.valueSum) {
            return a.valueSum > b.valueSum;
        }
        return a.action.bits < b.action.bits;
    });
    if (maxChildren >= 0 && static_cast<int>(order.size()) > maxChildren) {
        order.resize(static_cast<std::size_t>(maxChildren));
    }
    snapshot.children.reserve(order.size());
    for (const int edgeIndex : order) {
        const auto &edge = node->edges[edgeIndex];
        TreeEdgeInfo info;
        info.edgeIndex = edgeIndex;
        info.actionBits = edge.action.bits;
        info.visits = edge.visits;
        info.valueVisits = edge.valueVisits;
        info.childStates = static_cast<int>(edge.children.size());
        for (const auto &[childKey, child] : edge.children) {
            (void)childKey;
            if (child) {
                info.childVisits += child->visits;
            }
        }
        info.prior = edge.prior;
        info.value = edgeValueForSelection(edge, *node);
        info.sampleQ = edge.valueVisits > 0
            ? edge.valueSum / static_cast<double>(edge.valueVisits)
            : info.value;
        if (static_cast<std::size_t>(edgeIndex) < qDetails.raw.size()) {
            info.rawQ = qDetails.raw[static_cast<std::size_t>(edgeIndex)];
            info.normalizedQ = qDetails.normalized[
                static_cast<std::size_t>(edgeIndex)];
            info.completedQ = qDetails.scaled[
                static_cast<std::size_t>(edgeIndex)];
        }
        if (node == &root_ &&
            static_cast<std::size_t>(edgeIndex) < gumbelNoise_.size()) {
            const auto index = static_cast<std::size_t>(edgeIndex);
            info.gumbel = gumbelNoise_[index];
            info.gumbelScore = std::log(std::max(edge.prior, 1.0e-12)) +
                info.gumbel;
            info.completedScore = info.gumbelScore + info.completedQ;
            info.candidate = std::find(
                gumbelInitialEdges_.begin(), gumbelInitialEdges_.end(),
                edgeIndex) != gumbelInitialEdges_.end();
            info.activeCandidate = std::find(
                gumbelActiveEdges_.begin(), gumbelActiveEdges_.end(),
                edgeIndex) != gumbelActiveEdges_.end();
        } else {
            // Interior nodes do not receive independent Gumbel noise in Full
            // Gumbel MuZero.  Expose their deterministic improved-policy score
            // so the inspector can still explain the selected edge.
            info.completedScore = std::log(std::max(edge.prior, 1.0e-12)) +
                info.completedQ;
        }
        info.expandable = std::any_of(
            edge.children.begin(), edge.children.end(),
            [](const auto &entry) {
                return entry.second && !entry.second->edges.empty();
            });
        info.targeted = edge.targeted;
        info.randomSplit = edge.children.size() > 1;
        info.label = edge.label;
        snapshot.children.push_back(info);
    }
    return snapshot;
}

void NoslCombatTrainer::clearTable() {
    policy_.clear();
    values_.clear();
}

bool NoslCombatTrainer::saveTable(const std::string &path) const {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "# state_key action_bits probability value\n";
    for (const auto &[key, state] : policy_) {
        const auto valueIt = values_.find(key);
        const double value = valueIt == values_.end() ? 0.0 : valueIt->second;
        for (const auto &[action, probability] : state.probabilities) {
            output << key << ' ' << action << ' ' << probability << ' ' << value << '\n';
        }
    }
    return true;
}

}
