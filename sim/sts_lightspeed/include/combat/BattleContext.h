//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_BATTLECONTEXT_H
#define STS_LIGHTSPEED_BATTLECONTEXT_H

#include <vector>
#include <array>
#include <atomic>
#ifdef STS_ENABLE_RULE_TRACE
#include <functional>
#endif

#include "sts_common.h"

#include "data_structure/fixed_list.h"
#include "constants/Potions.h"
#include "constants/MonsterEncounters.h"
#include "combat/InputState.h"
#include "combat/Player.h"
#include "combat/Monster.h"
#include "combat/MonsterGroup.h"
#include "combat/ActionQueue.h"
#include "combat/CardQueue.h"
#include "combat/Actions.h"
#include "combat/CardManager.h"
#include "combat/CardSelectInfo.h"


namespace sts {



    enum class Outcome {
        UNDECIDED=0,
        PLAYER_VICTORY,
        PLAYER_LOSS,
        PLAYER_ESCAPE,
    };

    static constexpr const char * battleOutcomeStrings[] {
            "UNDECIDED",
            "PLAYER_VICTORY",
            "PLAYER_LOSS",
            "PLAYER_ESCAPE",
    };

    class GameContext;

    extern thread_local BattleContext *g_debug_bc;

    struct BattleContext {

        // begin for debugging purposes
        inline static std::atomic<int> sum{0}; // debug checksum, shared by native search threads
        bool haveUsedDiscoveryAction = false; // for tracking undefined behavior resulting from using the action
        bool undefinedBehaviorEvoked = false; // some cards cause inconsistent outcomes in games
        std::uint64_t seed = 0;
        int floorNum = 0;
        MonsterEncounter encounter = MonsterEncounter::INVALID;
        int loopCount = 0;
        int energyWasted = 0;
        int cardsDrawn = 0;
        // end for debugging purposes

        Random aiRng;
        Random cardRandomRng;
        Random miscRng;
        Random monsterHpRng;
        Random potionRng;
        Random shuffleRng;

        int ascension = 0;
        Outcome outcome = Outcome::UNDECIDED;
        InputState inputState = InputState::EXECUTING_ACTIONS;
        CardSelectInfo cardSelectInfo;

        int monsterTurnIdx = 6;

        bool isBattleOver = false;
        bool endTurnQueued = false;
        bool turnHasEnded = false;
        bool skipMonsterTurn = false;
        // Search control: stop after the complete enemy phase, before
        // next-turn start effects/draw/energy recharge.
        bool stopAfterEnemyTurn = false;
        // Observation marker set only after the stable boundary has actually
        // been reached.  Keep it separate from the control flag so in-flight
        // enemy actions are never mislabeled as value boundaries.
        bool postEnemyTurnBoundary = false;
        int pocketwatchCardsAtBoundary = 0;

        ActionQueue<50> actionQueue;
        CardQueue cardQueue;

        int potionCount = 0;
        int potionCapacity = 3;
        std::array<Potion, 5> potions;

        int turn = 0;
        Player player;
        MonsterGroup monsters;
        CardManager cards;

        // Acquisition order of the supported onUseCard relic callbacks.
        // Immutable during combat; copied with the battle and restored explicitly.
        std::vector<RelicId> onUseRelicOrder;
        // Actual acquisition order of supported onPlayerEndTurn callbacks.
        std::vector<RelicId> onEndTurnRelicOrder;

        CardQueueItem curCardQueueItem;

#ifdef STS_ENABLE_RULE_TRACE
        // Synchronous hook compiled only into the dedicated rule-data build.
        std::function<void(const BattleContext &)> atomicStepObserver;
#endif

        std::bitset<32> miscBits; // 0 stolen gold check,

        BattleContext() = default;
        BattleContext(const BattleContext &rhs) = default;

// ****************************************

        void init(const GameContext &gc);
        void init(const GameContext &gc, MonsterEncounter encounterToInit);

        void initRelics(const GameContext &gc);

        void exitBattle(GameContext &g) const;
        void updateRelicsOnExit(GameContext &g) const;
        void updateCardsOnExit(Deck &d) const; // for cards like ritual dagger, and eventually lesson learned results

// ****************************************

        void setRequiresStolenGoldCheck(bool value);
        [[nodiscard]] bool requiresStolenGoldCheck() const;
        [[nodiscard]] int getMonsterTurnNumber() const;  // returns 1 for first turn, 2 for second, etc.

// ****************************************

        void executeActions();

        void playCardQueueItem(CardQueueItem);
        void useCard();
        void useNoTriggerCard();

        void useAttackCard();
        void useSkillCard();
        void usePowerCard();

        void onUseAttackCard();
        void onUseSkillCard();
        void onUsePowerCard();

        void onUseStatusOrCurseCard();
        static bool hasOnUseCardHook(RelicId relic);
        static bool hasOnPlayerEndTurnHook(RelicId relic);
        void onUseCardRelics();
        void onAfterUseCard();

        void setState(InputState state);
        void addToTop(const Action &a);
        void addToBot(const Action &a);

        void addToTopCard(CardQueueItem item);
        void addToBotCard(CardQueueItem item);

        void checkCombat();
        void clearPostCombatActions();
        void cleanCardQueue();

        [[nodiscard]] bool isCardPlayAllowed() const;

        // **********************

        void endTurn(); // called when player clicks the end turn button
        void callEndOfTurnActions(); // GameActionManager.callEndTurnActions(), called when the end turn cardQueue item is reached.
        void onTurnEnding(); // AbstractRoom endTurn(), called when the turn actually ends.

        void callEndTurnEarlySequence(); // time eater

        void applyEndOfRoundPowers(); // game : MonsterGroup applyEndOfTurnPowers
        void enterPostEnemyTurnBoundary();
        void afterMonsterTurns();

        void obtainPotion(Potion p);
        void discardPotion(int idx);
        [[nodiscard]] bool canDrinkPotion(int idx) const;
        void drinkPotion(int idx, int target=0);

        void drawCards(int count);
        void discardAtEndOfTurn();
        void discardAtEndOfTurnHelper();

        void playTopCardInDrawPile(int monsterTargetIdx, bool exhausts);

        // A fresh AbstractCard.makeCopy, not a copy of an existing instance.
        // Capture combat-dependent base cost before any source-specific override.
        [[nodiscard]] CardInstance makeCardCopy(CardId id) const;
        void moveToHandHelper(CardInstance c);
        // Generated/returned-copy arrival, after identity and combat bookkeeping.
        // Unlike a plain transfer, ShowCardAndAddToHandEffect refreshes hand costs.
        void moveGeneratedCardToHand(CardInstance c);
        void exhaustSpecificCardInHand(int idx, std::int16_t uniqueId);
        void restoreRetainedCards(int count);
        void exhaustTopCardInHand();

        // this is
        void triggerOnEndOfTurnForPlayingCards();
        void triggerOnOtherCardPlayed(const CardInstance &usedCard);

        template <MonsterStatus s>
        void debuffEnemy(int idx, int amount, bool isSourceMonster=true);

        [[nodiscard]] int calculateCardDamage(const CardInstance &card, int targetIdx, int baseDamage) const;
        [[nodiscard]] DamageMatrix calculateCardDamageMatrix(const CardInstance &card, int baseDamage) const;
        [[nodiscard]] int calculateCardBlock(int baseBlock) const;

        void queuePurgeCard(const CardInstance &c, int target, bool inLimbo = true);
        void increaseCardSpecialData(std::int16_t uniqueId, int amount);
        void noOpRollMove(); // called by monsters to manipulate the aiRng counter when their rollMove function doesn't do anything

        void onManualDiscard(const CardInstance &c);
        void onCardDrawOrDiscard();
        void applyPowersToHand();
        void onShuffle();
        void triggerAndMoveToExhaustPile(CardInstance c);
        void mummifiedHandOnUsePower();

        // card select screens
        void openDiscoveryScreen(std::array<CardId, 3> discoveryCards, int copyCount);
        void openSimpleCardSelectScreen(CardSelectTask task, int count);

        // single card select helpers
        void chooseArmamentsCard(int handIdx);
        void chooseCodexCard(CardId id);
        void chooseDiscardToHandCard(int discardIdx, bool forZeroCost);
        bool chooseLiquidMemoriesCard(int discardIdx);
        void chooseDiscoveryCard(CardId id);
        void chooseDualWieldCard(int handIdx);
        void chooseExhaustOneCard(int handIdx);
        void chooseExhumeCard(int exhaustIdx);
        void chooseForethoughtCard(int handIdx);
        void chooseForethoughtCards(const fixed_list<int, 10> &idxs);
        void chooseHeadbuttCard(int discardIdx);
        void chooseRecycleCard(int handIdx);
        void chooseWarcryCard(int handIdx);

        // multi card helpers
        void chooseDrawToHandCards(const int *idxs, int cardCount);
        void chooseExhaustCards(const fixed_list<int,10> &idxs);
        void chooseGambleCards(const fixed_list<int,10> &idxs);

    };

    std::ostream& operator<<(std::ostream &os, const BattleContext &bc);

    template <PlayerStatus s>
    Action Actions::BuffPlayer(int amount) {
        return {[=] (BattleContext &bc) {
            const bool noDrawAlreadyActive = s == PS::NO_DRAW && bc.player.hasStatus<PS::NO_DRAW>();
            bc.player.buff<s>(amount);
            // ApplyPowerAction's repeated NoDraw branch returns before onModifyPower.
            if (!noDrawAlreadyActive) bc.applyPowersToHand();
        }};
    }

    template <PlayerStatus s>
    Action Actions::DebuffPlayer(int amount, bool isSourceMonster) {
        return {[=] (BattleContext &bc) {
            const bool immune = (s == PS::WEAK && bc.player.hasRelic<RelicId::GINGER>())
                || (s == PS::FRAIL && bc.player.hasRelic<RelicId::TURNIP>());
            bc.player.debuff<s>(amount, isSourceMonster);
            if (!immune) bc.applyPowersToHand();
        }};
    }

    template<PlayerStatus s>
    Action Actions::DecrementStatus(int amount) {
        return {[=] (BattleContext &bc) {
            const bool present = bc.player.hasStatus<s>();
            bc.player.decrementStatus<s>(amount);
            if (present) bc.applyPowersToHand();
        }};
    }

    template <PlayerStatus s>
    Action Actions::RemoveStatus() {
        return {[=] (BattleContext &bc) {
            const bool present = bc.player.hasStatus<s>();
            bc.player.setHasStatus<s>(false);
            if (present) bc.applyPowersToHand();
        }};
    }

    template<MonsterStatus s>
    Action Actions::BuffEnemy(int idx, int amount) {
        return {[=] (BattleContext &bc) {
            // ApplyPowerAction checks again when the queued action executes.
            // Earlier damage may have killed or knocked down this target.
            auto &monster = bc.monsters.arr[idx];
            if (!monster.isTargetable()) return;
            monster.buff<s>(amount);
            bc.applyPowersToHand();
        }};
    }

    template <MonsterStatus s>
    Action Actions::DebuffEnemy(int idx, int amount, bool isSourceMonster) {
        return {[=] (BattleContext &bc) {
            bc.debuffEnemy<s>(idx, amount, isSourceMonster);
        }};
    }

    template <MonsterStatus s>
    Action Actions::DebuffAllEnemy(int amount, bool isSourceMonster) {
        // todo this should just add all to bot immediately, not be called first
        // ^^ never mind i think adding to top is a workaround here

        return {[=] (BattleContext &bc) {
            for (int i = bc.monsters.monsterCount-1; i >= 0; --i) {
                if (bc.monsters.arr[i].isTargetable()) {
                    bc.addToTop(Actions::DebuffEnemy<s>(i, amount, isSourceMonster));
                }
            }
        }};
    }

    template <MonsterStatus s>
    void BattleContext::debuffEnemy(int idx, int amount, bool isSourceMonster) {
        // todo poison and snake skull
        Monster &e = monsters.arr[idx];
        if (!e.isTargetable()) return;

        if (e.hasStatus<MS::ARTIFACT>()) {
            e.decrementStatus<MS::ARTIFACT>();
            applyPowersToHand(); // Artifact's queued ReducePower/RemovePower hook.
            return;
        }

        e.addDebuff<s>(amount, isSourceMonster);
        applyPowersToHand();
        if (player.hasStatus<PS::SADISTIC>()) {
            addToBot( Actions::DamageEnemy(idx, player.getStatus<PS::SADISTIC>()) );
        }

        if (s == MS::VULNERABLE && player.hasRelic<RelicId::CHAMPION_BELT>()) {
            addToBot(Actions::DebuffEnemy<MS::WEAK>(idx, 1, false) );
        }
    }


}

#endif //STS_LIGHTSPEED_BATTLECONTEXT_H
