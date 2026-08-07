//
// Created by keega on 9/16/2021.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"

#include "slaythespire.h"

// added for combat bindings (diff testing / RL)
#include <memory>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include "sim/search/GameAction.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include "constants/Cards.h"
#include "constants/MonsterEncounters.h"


using namespace sts;

// ===== combat-state export + action helpers (added for diff testing / RL) =====
// The C++ side only emits structured RAW data keyed by the sim's own enum-name
// strings; mapping those to the real game's ids is done in Python (schemas/),
// symmetric with the real-game normalizer. Keeps the binding minimal.
namespace {
namespace py = pybind11;

template <typename Pile>
py::list cardPileToList(const Pile &pile) {
    py::list out;
    for (const auto &c : pile) {
        out.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(c.getId())], c.getUpgradeCount()));
    }
    return out;
}

py::list playerPowersToList(const sts::Player &p) {
    py::list out;
    if (p.strength)  out.append(py::make_tuple("Strength", p.strength));
    if (p.dexterity) out.append(py::make_tuple("Dexterity", p.dexterity));
    if (p.focus)     out.append(py::make_tuple("Focus", p.focus));
    if (p.artifact)  out.append(py::make_tuple("Artifact", p.artifact));
    for (const auto &kv : p.statusMap) {
        if (kv.second != 0) {
            out.append(py::make_tuple(playerStatusStrings[static_cast<int>(kv.first)], kv.second));
        }
    }
    return out;
}

py::list monsterPowersToList(const sts::Monster &m) {
    py::list out;
    for (int s = 0; s < static_cast<int>(sts::MonsterStatus::INVALID); ++s) {
        const int v = m.getStatusInternal(static_cast<sts::MonsterStatus>(s));
        if (v != 0) {
            out.append(py::make_tuple(sts::enemyStatusStrings[s], v));
        }
    }
    return out;
}

py::dict rawCombatState(const sts::BattleContext &bc) {
    py::dict d;
    d["turn"] = bc.turn;
    d["outcome"] = static_cast<int>(bc.outcome);
    d["input_state"] = static_cast<int>(bc.inputState);

    py::dict player;
    player["hp"] = bc.player.curHp;
    player["max_hp"] = bc.player.maxHp;
    player["block"] = bc.player.block;
    player["energy"] = bc.player.energy;
    player["powers"] = playerPowersToList(bc.player);
    d["player"] = player;

    py::list monsters;
    for (int i = 0; i < bc.monsters.monsterCount; ++i) {
        const auto &m = bc.monsters.arr[i];
        py::dict md;
        md["id"] = sts::monsterIdStrings[static_cast<int>(m.id)];
        md["hp"] = m.curHp;
        md["max_hp"] = m.maxHp;
        md["block"] = m.block;
        md["gone"] = m.isDeadOrEscaped();
        md["half_dead"] = m.halfDead;
        // current intent: the move the monster has rolled for this turn. setMove
        // shifts the previous move to moveHistory[1] and stores the new one in
        // moveHistory[0], so [0] is the *current* intent (the obs MOVE_VOCAB id).
        const auto curMove = m.moveHistory[0];
        md["move_id"] = static_cast<int>(curMove);
        md["move"] = sts::monsterMoveStrings[static_cast<int>(curMove)];
        const auto dmg = m.getMoveBaseDamage(bc);
        md["base_damage"] = dmg.damage;
        // Use the simulator's actual damage pipeline for the displayed intent.
        // This includes Strength, Weak/Paper Krane, player Vulnerable/Odd
        // Mushroom, Surrounded, Wrath, Intangible, and the game's rounding.
        md["adjusted_damage"] = dmg.damage > 0
            ? m.calculateDamageToPlayer(bc, dmg.damage)
            : 0;
        md["hits"] = dmg.attackCount;
        md["powers"] = monsterPowersToList(m);
        monsters.append(md);
    }
    d["monsters"] = monsters;

    py::list hand;
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        const auto &c = bc.cards.hand[i];
        const int currentCost = c.freeToPlayOnce ? 0 : c.costForTurn;
        hand.append(py::make_tuple(
            sts::cardEnumStrings[static_cast<int>(c.getId())],
            c.getUpgradeCount(),
            currentCost
        ));
    }
    d["hand"] = hand;
    d["draw"] = cardPileToList(bc.cards.drawPile);
    d["discard"] = cardPileToList(bc.cards.discardPile);
    d["exhaust"] = cardPileToList(bc.cards.exhaustPile);
    return d;
}

std::vector<sts::search::Action> legalCombatActions(const sts::BattleContext &bc) {
    using SA = sts::search::Action;
    using AT = sts::search::ActionType;
    std::vector<SA> out;
    if (bc.outcome != sts::Outcome::UNDECIDED) {
        return out;
    }
    switch (bc.inputState) {
        case sts::InputState::PLAYER_NORMAL:
            if (bc.isCardPlayAllowed()) {
                for (int h = 0; h < bc.cards.cardsInHand; ++h) {
                    const auto &c = bc.cards.hand[h];
                    if (!c.canUseOnAnyTarget(bc)) {
                        continue;
                    }
                    if (c.requiresTarget()) {
                        for (int t = 0; t < bc.monsters.monsterCount; ++t) {
                            if (bc.monsters.arr[t].isTargetable()) {
                                out.push_back(SA(AT::CARD, h, t));
                            }
                        }
                    } else {
                        out.push_back(SA(AT::CARD, h));
                    }
                }
            }
            out.push_back(SA(AT::END_TURN));
            break;
        case sts::InputState::CARD_SELECT:
            for (const auto &a : SA::enumerateCardSelectActions(bc)) {
                out.push_back(a);
            }
            break;
        default:
            break;
    }
    return out;
}

// Authoritative legal IN-COMBAT potion actions, from the sim's own per-Action
// validator (search::Action::isValidAction -> isValidPotionAction). We don't
// hand-roll potion legality: we probe each held slot with the exact POTION
// Action the env would later execute and keep only the ones the sim accepts.
// Returns a list of {slot, requires_target, targets} dicts:
//   * requires_target == false -> drink in one step (drink_potion(slot)); targets
//     is empty.
//   * requires_target == true  -> the potion needs a monster target (Fire/Fear/
//     Poison/Weak); targets lists the monster indices the sim accepts (each probed
//     with the real POTION Action, so it already excludes dead/untargetable foes).
//     The thin layer runs its PICK_TARGET substep over exactly these, then calls
//     drink_potion(slot, target). potionRequiresTarget() (Potions.h) is the sim's
//     own notion of targeting, so we surface it rather than re-deriving it.
// Discard-only potions (Fairy etc., or a no-effect drink) are intentionally NOT
// offered: in combat the only valid drink is what isValidPotionAction accepts.
py::list legalCombatPotions(const sts::BattleContext &bc) {
    using SA = sts::search::Action;
    using AT = sts::search::ActionType;
    py::list out;
    if (bc.outcome != sts::Outcome::UNDECIDED ||
        bc.inputState != sts::InputState::PLAYER_NORMAL) {
        return out;
    }
    for (int slot = 0; slot < bc.potionCapacity; ++slot) {
        const auto p = bc.potions[slot];
        if (p == sts::Potion::INVALID || p == sts::Potion::EMPTY_POTION_SLOT) {
            continue;
        }
        const bool needsTarget = sts::potionRequiresTarget(p);
        py::list targets;
        if (needsTarget) {
            // a targeted potion (Fire/Fear/Poison/Weak): the env enumerates one
            // action per accepted monster, mirroring the targeted card-play path,
            // so we list exactly the monster indices the sim's validator accepts.
            for (int t = 0; t < bc.monsters.monsterCount; ++t) {
                if (SA(AT::POTION, slot, t).isValidAction(bc)) {
                    targets.append(t);
                }
            }
            if (py::len(targets) == 0) continue;   // no legal target -> not usable
        } else if (!SA(AT::POTION, slot, 0).isValidAction(bc)) {
            // untargeted but the sim rejects drinking it (e.g. Fairy in Battle,
            // or a potion with no in-combat drink effect) -> not offered.
            continue;
        }
        py::dict d;
        d["slot"] = slot;
        d["requires_target"] = needsTarget;
        d["targets"] = targets;
        out.append(d);
    }
    return out;
}

// Authoritative legal OUT-OF-COMBAT (overworld) potion actions, from the sim's
// own GameAction potion validator (isValidPotionAction). Only a handful of
// potions do anything outside combat (Blood/Entropic Brew/Fruit Juice drink;
// the rest are discard-only and we do NOT surface a discard verb to the agent).
// A targeted potion never applies out of combat (no monsters), so every
// overworld entry is single-step (requires_target is always false here). We
// construct the GameAction with the potion bit set (0x80000000) so isValidAction
// routes to isValidPotionAction; the discard bit (0x40000000) is left clear so we
// probe the DRINK path specifically.
py::list legalOverworldPotions(const sts::GameContext &gc) {
    py::list out;
    if (gc.outcome != sts::GameOutcome::UNDECIDED) {
        return out;
    }
    for (int slot = 0; slot < gc.potionCapacity; ++slot) {
        const auto p = gc.potions[slot];
        if (p == sts::Potion::INVALID || p == sts::Potion::EMPTY_POTION_SLOT) {
            continue;
        }
        // bits: potion-action flag | idx1=slot. discard bit clear == DRINK probe.
        sts::search::GameAction a(0x80000000U | (slot & 0xFF));
        if (!a.isValidAction(gc)) {
            continue;
        }
        py::dict d;
        d["slot"] = slot;
        d["requires_target"] = false;   // no targets exist out of combat
        d["targets"] = py::list();
        out.append(d);
    }
    return out;
}

std::unique_ptr<sts::BattleContext> createBattle(const sts::GameContext &gc, const std::string &encounterName) {
    auto enc = sts::MonsterEncounter::INVALID;
    constexpr int n = sizeof(sts::monsterEncounterStrings) / sizeof(sts::monsterEncounterStrings[0]);
    for (int i = 0; i < n; ++i) {
        if (encounterName == sts::monsterEncounterStrings[i]) {
            enc = static_cast<sts::MonsterEncounter>(i);
            break;
        }
    }
    auto bc = std::make_unique<sts::BattleContext>();
    bc->init(gc, enc);
    return bc;
}

std::vector<std::string> encounterNames() {
    std::vector<std::string> v;
    constexpr int n = sizeof(sts::monsterEncounterStrings) / sizeof(sts::monsterEncounterStrings[0]);
    for (int i = 1; i < n; ++i) {  // skip INVALID
        v.emplace_back(sts::monsterEncounterStrings[i]);
    }
    return v;
}

// ===== out-of-combat introspection (added for full-run replay / diff) =====
// Symmetric with the combat side: emit raw data keyed by the sim's own enum
// strings (cards) or the game's stable ids (relics), mapped in Python.

py::list deckIds(const sts::GameContext &gc) {
    py::list out;
    for (const auto &c : gc.deck.cards) {
        out.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(c.getId())], c.getUpgraded()));
    }
    return out;
}

// sts::relicIds[] holds the game's stable ids (e.g. "Art of War"), matching the
// real CommunicationMod relic `id` field -- not relicNames[] (display strings).
py::list gcRelicIds(const sts::GameContext &gc) {
    py::list out;
    for (const auto &r : gc.relics.relics) {
        out.append(std::string(sts::relicIds[static_cast<int>(r.id)]));
    }
    return out;
}

// the cards offered on a CARD_SELECT (grid) screen, in the sim's order
py::list selectCards(const sts::GameContext &gc) {
    py::list out;
    for (const auto &sc : gc.info.toSelectCards) {
        out.append(py::make_tuple(sts::cardEnumStrings[static_cast<int>(sc.card.getId())], sc.card.getUpgraded()));
    }
    return out;
}

// reverse lookups: real-side strings -> sim enums (for force-syncing gc state
// to a real-run snapshot before a combat, decoupled from out-of-combat RNG).
sts::CardId cardIdFromEnum(const std::string &name) {
    constexpr int n = sizeof(sts::cardEnumStrings) / sizeof(sts::cardEnumStrings[0]);
    for (int i = 0; i < n; ++i) {
        if (name == sts::cardEnumStrings[i]) return static_cast<sts::CardId>(i);
    }
    throw std::runtime_error("unknown card enum: " + name);
}

sts::RelicId relicIdFromString(const std::string &name) {
    constexpr int n = sizeof(sts::relicIds) / sizeof(sts::relicIds[0]);
    for (int i = 0; i < n; ++i) {
        if (name == sts::relicIds[i]) return static_cast<sts::RelicId>(i);
    }
    throw std::runtime_error("unknown relic id: " + name);
}

void setDeck(sts::GameContext &gc, const std::vector<std::pair<std::string, int>> &cards) {
    gc.deck.cards.clear();
    gc.deck.cardTypeCounts = {0, 0, 0, 0};
    gc.deck.upgradeableCount = 0;
    gc.deck.transformableCount = 0;
    for (const auto &[name, up] : cards) {
        gc.deck.obtainRaw(sts::Card(cardIdFromEnum(name), up));
    }
}

void setRelics(sts::GameContext &gc, const std::vector<std::string> &ids) {
    gc.relics.relics.clear();
    gc.relics.relicBits0 = gc.relics.relicBits1 = gc.relics.relicBits2 = 0;
    for (const auto &id : ids) {
        gc.relics.add(sts::RelicInstance{relicIdFromString(id), 0});
    }
}

// The sim's OWN out-of-combat action enumerator (search::GameAction) already
// solves legality for every overworld screen -- events (incl. phase-dependent
// ones like Golden Idol / Cursed Tome), campfires (relic/key-gated), shop,
// rewards, card-select -- with the same relic/gold/deck awareness the search
// agent uses. We expose just the option *indices* the model needs per screen,
// keyed the same way the thick layer already dispatches (chooseEventOption(i),
// chooseCampfireOption(i)). This is the out-of-combat analog of the bound
// combat `legalCombatActions`, and the authoritative fix for the event stub.

// valid event-option indices for the CURRENT event (getIdx1 of each action).
py::list legalEventOptions(const sts::GameContext &gc) {
    py::list out;
    for (const auto &a : sts::search::GameAction::getAllActionsInState(gc)) {
        out.append(a.getIdx1());
    }
    return out;
}

// valid campfire-option indices (0=rest,1=smith,2=recall,3=lift,4=toke,5=dig,
// 6=skip) for the current REST_ROOM, gated by relics/keys exactly as the sim's
// own chooseCampfireOption expects.
py::list legalCampfireOptions(const sts::GameContext &gc) {
    py::list out;
    for (const auto &a : sts::search::GameAction::getAllActionsInState(gc)) {
        out.append(a.getIdx1());
    }
    return out;
}

// what is claimable on the current combat-reward screen
py::dict rewardInfo(const sts::GameContext &gc) {
    const auto &r = gc.info.rewardsContainer;
    py::dict d;
    py::list gold, relics, potions;
    for (int i = 0; i < r.goldRewardCount; ++i) gold.append(r.gold[i]);
    for (int i = 0; i < r.relicCount; ++i) relics.append(std::string(sts::relicIds[static_cast<int>(r.relics[i])]));
    for (int i = 0; i < r.potionCount; ++i) potions.append(std::string(sts::getPotionName(r.potions[i])));
    d["gold"] = gold;
    d["relics"] = relics;
    d["potions"] = potions;
    d["card_count"] = r.cardRewardCount;
    return d;
}

// The current shop's shelf, read POSITIONALLY (slot index is the buy key the
// shop_buy_* methods expect): cards[7], relics[3], potions[3]. A price of -1
// marks a sold/empty slot (Shop::cardPrice etc. return -1 after a purchase).
// Ids match the rest of the bindings: cardEnumStrings / relicIds / getPotionName.
py::dict shopContents(const sts::GameContext &gc) {
    const auto &s = gc.info.shop;
    py::dict d;
    py::list cards, relics, potions;
    for (int i = 0; i < 7; ++i) {
        py::dict c;
        c["id"] = std::string(sts::cardEnumStrings[static_cast<int>(s.cards[i].getId())]);
        c["upgrades"] = s.cards[i].getUpgraded();
        c["price"] = s.cardPrice(i);
        cards.append(c);
    }
    for (int i = 0; i < 3; ++i) {
        py::dict r;
        r["id"] = std::string(sts::relicIds[static_cast<int>(s.relics[i])]);
        r["price"] = s.relicPrice(i);
        relics.append(r);
    }
    for (int i = 0; i < 3; ++i) {
        py::dict p;
        p["id"] = std::string(sts::getPotionName(s.potions[i]));
        p["price"] = s.potionPrice(i);
        potions.append(p);
    }
    d["cards"] = cards;
    d["relics"] = relics;
    d["potions"] = potions;
    d["remove_cost"] = s.removeCost;
    return d;
}

// Authoritative legal shop options, from the sim's own getAllShopActions (gold-
// and sold-out-gated). Unlike legalEventOptions, idx1 COLLIDES across kinds
// (card#0 / relic#0 / potion#0 all have idx1==0), so we return (kind, idx)
// pairs. kind is a stable string keyed to shop_buy_*/shop_remove/shop_proceed:
// "card"/"relic"/"potion"/"remove"/"skip". "skip" always present.
py::list legalShopOptions(const sts::GameContext &gc) {
    using RAT = sts::search::GameAction::RewardsActionType;
    py::list out;
    for (const auto &a : sts::search::GameAction::getAllActionsInState(gc)) {
        const char *kind;
        switch (a.getRewardsActionType()) {
            case RAT::CARD:        kind = "card";   break;
            case RAT::RELIC:       kind = "relic";  break;
            case RAT::POTION:      kind = "potion"; break;
            case RAT::CARD_REMOVE: kind = "remove"; break;
            case RAT::SKIP:        kind = "skip";   break;
            default:               continue;
        }
        out.append(py::make_tuple(kind, a.getIdx1()));
    }
    return out;
}

// The OVERWORLD card-select (grid) screen's semantics + progress, for the obs
// select_context / pick_remaining / selected fields (obs_design §4.1). Read off
// gc.info (ScreenStateInfo): selectScreenType says WHAT this grid is for
// (TRANSFORM/UPGRADE/REMOVE/...), toSelectCount is how many must be picked, and
// haveSelectedCards are the ones already picked THIS screen (the sim removes a
// picked card from toSelectCards and pushes it here, so already-picked cards are
// NOT in get_select_cards() anymore -- we surface their ids here instead).
const char *selectScreenTypeStr(sts::CardSelectScreenType t) {
    using T = sts::CardSelectScreenType;
    switch (t) {
        case T::TRANSFORM:         return "TRANSFORM";
        case T::TRANSFORM_UPGRADE: return "TRANSFORM_UPGRADE";
        case T::UPGRADE:           return "UPGRADE";
        case T::REMOVE:            return "REMOVE";
        case T::DUPLICATE:         return "DUPLICATE";
        case T::OBTAIN:            return "OBTAIN";
        case T::BOTTLE:            return "BOTTLE";
        case T::BONFIRE_SPIRITS:   return "BONFIRE_SPIRITS";
        case T::INVALID:
        default:                   return "INVALID";
    }
}

py::dict selectScreenInfo(const sts::GameContext &gc) {
    const auto &info = gc.info;
    py::dict d;
    d["context"] = std::string(selectScreenTypeStr(info.selectScreenType));
    d["pick_count"] = info.toSelectCount;
    d["have_selected"] = static_cast<int>(info.haveSelectedCards.size());
    py::list sel;
    for (const auto &sc : info.haveSelectedCards) {
        sel.append(std::string(sts::cardEnumStrings[static_cast<int>(sc.card.getId())]));
    }
    d["selected"] = sel;
    return d;
}

// The COMBAT card-select sub-screen's semantics + progress (bc.cardSelectInfo),
// for the same obs fields while in combat (Armaments/Headbutt/Gamble/...).
//
// Two structurally different combat card-selects exist:
//   * SINGLE (Armaments/Headbutt/...) -- one pick resolves the screen in one
//     action; the thin layer maps each candidate to a PICK_CARD id.
//   * MULTI  (Exhaust Many / Gamble)  -- the sim resolves the WHOLE screen with
//     one Action whose low 10 bits are a hand-index bitset (getSelectedIdxs() ->
//     chooseExhaustCards/chooseGambleCards). The thin layer therefore TOGGLES
//     hand indices across env steps and CONFIRMs once, building that bitset.
// `multi` lets the thin layer pick which machine to drive. For a MULTI screen we
// also expose `cards_in_hand` (the toggleable indices 0..n-1) and `pick_max`
// (the cap: Exhaust Many = pickCount limit; Gamble = the whole hand). pickCount
// is only meaningful for MULTI -- SINGLE tasks leave it 0 (see CardSelectInfo).
py::dict combatSelectInfo(const sts::BattleContext &bc) {
    const auto &csi = bc.cardSelectInfo;
    const auto task = csi.cardSelectTask;
    const bool multi = task == sts::CardSelectTask::EXHAUST_MANY ||
                       task == sts::CardSelectTask::GAMBLE;
    py::dict d;
    d["context"] = std::string(
        sts::cardSelectTaskStrings[static_cast<int>(task)]);
    d["pick_count"] = csi.pickCount;
    d["can_pick_zero"] = csi.canPickZero;
    d["can_pick_any"] = csi.canPickAnyNumber;
    d["multi"] = multi;
    d["cards_in_hand"] = bc.cards.cardsInHand;
    // pick_max: hand indices a MULTI screen may toggle. Exhaust Many caps at its
    // limit (pickCount); Gamble has no limit beyond the hand. 0 for SINGLE.
    int pickMax = 0;
    if (task == sts::CardSelectTask::EXHAUST_MANY) {
        pickMax = csi.pickCount;
    } else if (task == sts::CardSelectTask::GAMBLE) {
        pickMax = bc.cards.cardsInHand;
    }
    d["pick_max"] = pickMax;
    return d;
}

} // namespace

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");

    pybind11::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    pybind11::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def("playout", &search::ScumSearchAgent2::playout);

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def("choose_neow_option",
             [](GameContext &gc, int idx) { gc.chooseNeowOption(gc.info.neowRewards[idx]); },
             "apply the Neow reward option at the given index (0-3)")
        .def("transition_to_map_node",
             [](GameContext &gc, int x) { gc.transitionToMapNode(x); },
             "advance to the map node at column x on the next row (enters that room)")
        .def("_dbg_set_card_rng", [](GameContext &gc, int n) { gc.cardRng = sts::Random(gc.seed, n); })
        .def("_dbg_card_reward", [](GameContext &gc) {
            auto cr = gc.createCardReward(sts::Room::MONSTER);
            std::vector<std::pair<std::string,int>> out;
            for (auto &c : cr) out.emplace_back(sts::cardEnumStrings[(int)c.getId()], c.getUpgraded());
            return out;
        })
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        // ----- out-of-combat replay/diff (added for full-run replay) -----
        .def("deck_ids", &deckIds, "deck as [(sim_card_enum, upgrade_count), ...]")
        .def("relic_ids", &gcRelicIds, "relics as [stable_id_string, ...]")
        .def("set_deck", &setDeck, "replace the deck with [(sim_card_enum, upgrade_count), ...]")
        .def("set_relics", &setRelics, "replace the relics with [stable_id_string, ...]")
        .def_property_readonly("screen_state_name",
               [](const GameContext &gc) { return std::string(pybind11::str(pybind11::cast(gc.screenState))); })
        .def_property_readonly("cur_event_name",
               [](const GameContext &gc) { return std::string(sts::eventIdStrings[static_cast<int>(gc.curEvent)]); })
        .def_property_readonly("burning_elite_x", [](const GameContext &gc) { return gc.map ? gc.map->burningEliteX : -1; })
        .def_property_readonly("burning_elite_y", [](const GameContext &gc) { return gc.map ? gc.map->burningEliteY : -1; })
        // decision dispatch (mirrors GameContext's own screen state machine)
        .def("choose_event_option", [](GameContext &gc, int i) { gc.chooseEventOption(i); })
        .def("choose_campfire_option", [](GameContext &gc, int i) { gc.chooseCampfireOption(i); })
        .def("legal_event_options", &legalEventOptions,
             "valid option indices for the current event (sim-authoritative, phase/relic aware)")
        .def("legal_campfire_options", &legalCampfireOptions,
             "valid campfire option indices (0=rest,1=smith,2=recall,3=lift,4=toke,5=dig,6=skip)")
        .def("choose_select_card_option", [](GameContext &gc, int i) { gc.chooseSelectCardScreenOption(i); })
        .def("choose_treasure_room", [](GameContext &gc, bool open) { gc.chooseTreasureRoomOption(open); })
        .def("choose_boss_relic", [](GameContext &gc, int i) { gc.chooseBossRelic(i); })
        .def("get_select_cards", &selectCards, "cards offered on the current CARD_SELECT/grid screen")
        .def("select_screen_info", &selectScreenInfo,
             "overworld card-select screen semantics+progress: {context (CardSelectScreenType "
             "name), pick_count, have_selected, selected:[card ids already picked this screen]}")
        // combat-reward screen
        .def("get_reward_info", &rewardInfo, "claimable items on the current REWARDS screen")
        .def("claim_gold", [](GameContext &gc, int idx) { auto &r = gc.info.rewardsContainer; gc.obtainGold(r.gold[idx]); r.removeGoldReward(idx); })
        .def("claim_relic", [](GameContext &gc, int idx) {
            auto &r = gc.info.rewardsContainer;
            // Some relics open a screen when obtained (e.g. Bottled Tornado's
            // "pick a Power to bottle"). That screen's confirm calls
            // regainControl(), which would otherwise run -- and consume -- the
            // combat's return-to-map action, leaving it null and crashing the
            // later PROCEED. Redirect the screen's regainControl back to the
            // REWARDS screen (more rewards may remain) and restore the original
            // action so the final PROCEED still returns to the map.
            auto saved = gc.regainControlAction;
            gc.regainControlAction = [saved](GameContext &g) {
                g.screenState = ScreenState::REWARDS;
                g.regainControlAction = saved;
            };
            bool opened = gc.obtainRelic(r.relics[idx]);
            r.removeRelicReward(idx);
            if (!opened) gc.regainControlAction = saved;
        })
        .def("claim_potion", [](GameContext &gc, int idx) { auto &r = gc.info.rewardsContainer; gc.obtainPotion(r.potions[idx]); r.removePotionReward(idx); })
        .def("discard_potion", [](GameContext &gc, int slot) {
            if (slot >= 0 && slot < gc.potionCapacity && gc.potions[slot] != sts::Potion::EMPTY_POTION_SLOT) {
                gc.potions[slot] = sts::Potion::EMPTY_POTION_SLOT;
                --gc.potionCount;
            }
        }, "discard the held potion in slot idx (0-based), out of combat")
        .def("drink_potion", [](GameContext &gc, int slot) { gc.drinkPotionAtIdx(slot); },
             pybind11::arg("idx"),
             "drink the held potion in slot idx (0-based) OUT OF COMBAT (Blood/Entropic "
             "Brew/Fruit Juice etc.); use BattleContext.drink_potion in combat")
        .def("legal_overworld_potions", &legalOverworldPotions,
             "authoritative legal out-of-combat potion DRINKS as [{slot, requires_target}]; "
             "requires_target is always false out of combat (no targets). sim-gated by "
             "GameAction::isValidPotionAction (only a few potions drink outside combat)")
        .def_property_readonly("overworld_potions", [](const GameContext &gc) {
            pybind11::list lst;
            for (int i = 0; i < gc.potionCapacity; ++i) lst.append(std::string(sts::getPotionName(gc.potions[i])));
            return lst;
        }, "held potion slots out of combat")
        .def("reward_proceed", [](GameContext &gc) { gc.regainControl(); })
        // shop
        .def("shop_buy_card", [](GameContext &gc, int idx) { gc.info.shop.buyCard(gc, idx); })
        .def("shop_buy_relic", [](GameContext &gc, int idx) { gc.info.shop.buyRelic(gc, idx); })
        .def("shop_buy_potion", [](GameContext &gc, int idx) { gc.info.shop.buyPotion(gc, idx); })
        .def("shop_remove", [](GameContext &gc) { gc.info.shop.buyCardRemove(gc); })
        .def("shop_proceed", [](GameContext &gc) { gc.screenState = ScreenState::MAP_SCREEN; })
        .def("shop_contents", &shopContents,
             "the current shop shelf, positional: {cards:[{id,upgrades,price}x7], "
             "relics:[{id,price}x3], potions:[{id,price}x3], remove_cost}. price -1 == sold/empty")
        .def("legal_shop_options", &legalShopOptions,
             "authoritative legal shop actions as (kind, idx) pairs; kind in "
             "{card,relic,potion,remove,skip} (sim-gated by gold/sold-out)")
        // obs globals: character + ascension are construction-time, boss_relic_ids
        // are the 3 choices offered on a BOSS_RELIC_REWARDS screen (info.bossRelics).
        .def_property_readonly("character",
            [](const GameContext &gc) {
                switch (gc.cc) {
                    case CharacterClass::IRONCLAD: return "IRONCLAD";
                    case CharacterClass::SILENT:   return "SILENT";
                    case CharacterClass::DEFECT:   return "DEFECT";
                    case CharacterClass::WATCHER:  return "WATCHER";
                    default:                       return "INVALID";
                }
            }, "the player's character class as a stable enum-name string")
        .def_readwrite("ascension", &GameContext::ascension)
        .def("boss_relic_ids", [](const GameContext &gc) {
            pybind11::list out;
            for (int i = 0; i < 3; ++i)
                out.append(std::string(sts::relicIds[static_cast<int>(gc.info.bossRelics[i])]));
            return out;
        }, "the 3 boss relics offered on the current boss-relic screen, as stable ids")
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        .def_readwrite("cur_room", &GameContext::curRoom)
//        .def_readwrite("cur_event", &GameContext::curEvent) // todo standardize event names
        .def_readwrite("boss", &GameContext::boss)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard);

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    pybind11::class_<Map> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int,int,bool>());
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState");
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room");
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<CardId>(m, "CardId")
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter");
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId");
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    // ===== combat bindings (added for diff testing / RL) =====
    pybind11::enum_<search::ActionType>(m, "ActionType")
        .value("CARD", search::ActionType::CARD)
        .value("POTION", search::ActionType::POTION)
        .value("SINGLE_CARD_SELECT", search::ActionType::SINGLE_CARD_SELECT)
        .value("MULTI_CARD_SELECT", search::ActionType::MULTI_CARD_SELECT)
        .value("END_TURN", search::ActionType::END_TURN);

    pybind11::class_<search::Action>(m, "Action")
        .def(pybind11::init<search::ActionType>())
        .def(pybind11::init<search::ActionType, int>())
        .def(pybind11::init<search::ActionType, int, int>())
        .def_property_readonly("action_type", &search::Action::getActionType)
        .def_property_readonly("source_idx", &search::Action::getSourceIdx)
        .def_property_readonly("target_idx", &search::Action::getTargetIdx)
        .def("__repr__", [](const search::Action &a) {
            std::ostringstream oss;
            oss << "<Action type=" << static_cast<int>(a.getActionType())
                << " src=" << a.getSourceIdx() << " tgt=" << a.getTargetIdx() << ">";
            return oss.str();
        });

    pybind11::class_<BattleContext>(m, "BattleContext")
        .def("to_dict", &rawCombatState, "export the combat state as a raw dict (sim-side, normalized to canonical in Python)")
        .def("legal_actions", &legalCombatActions, "list the legal actions in the current state")
        .def("legal_potions", &legalCombatPotions,
             "authoritative legal in-combat potion uses as [{slot, requires_target, targets}]; "
             "requires_target true == needs a monster target (Fire/Fear/Poison/Weak) and targets "
             "lists the accepted monster idxs (drink via drink_potion(slot, target)); false == "
             "drink in one step (targets empty). sim-gated by search::Action::isValidPotionAction")
        .def("select_info", &combatSelectInfo,
             "combat card-select sub-screen semantics+progress: {context (CardSelectTask name), "
             "pick_count, can_pick_zero, can_pick_any}")
        .def("execute", [](BattleContext &bc, const search::Action &a) { a.execute(bc); }, "apply an action and pump to the next input state")
        .def_property_readonly("turn", [](const BattleContext &bc) { return bc.turn; })
        .def_property_readonly("outcome", [](const BattleContext &bc) { return static_cast<int>(bc.outcome); })
        .def_property_readonly("input_state", [](const BattleContext &bc) { return static_cast<int>(bc.inputState); })
        .def_property_readonly("is_battle_over", [](const BattleContext &bc) { return bc.outcome != Outcome::UNDECIDED; })
        .def("exit_battle", [](const BattleContext &bc, GameContext &gc) { bc.exitBattle(gc); },
             "write combat results back into the GameContext (hp/gold/deck/rng) and open the reward screen on victory")
        .def("drink_potion", [](BattleContext &bc, int idx, int target) { bc.drinkPotion(idx, target); },
             pybind11::arg("idx"), pybind11::arg("target") = 0, "use the potion in slot idx (0-based) on target monster idx")
        .def("discard_potion", [](BattleContext &bc, int idx) { bc.discardPotion(idx); }, "discard the potion in slot idx (0-based)")
        .def_property_readonly("potions", [](const BattleContext &bc) {
            pybind11::list lst;
            for (int i = 0; i < bc.potionCapacity; ++i) lst.append(std::string(sts::getPotionName(bc.potions[i])));
            return lst;
        }, "combat potion slots");

    m.def("create_battle", &createBattle, "construct a combat from a GameContext + encounter name (forces the encounter)");
    m.def("encounter_names", &encounterNames, "list valid encounter names for create_battle");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")


