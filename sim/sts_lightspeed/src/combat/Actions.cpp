//
// Created by gamerpuppy on 7/4/2021.
//

#include <algorithm>
#include <limits>
#include <stdexcept>
#include "combat/Actions.h"
#include "combat/BattleContext.h"
#include "game/Game.h"

using namespace sts;

Action Actions::SetState(InputState state) {
    return {[=] (BattleContext &bc) {
        bc.setState(state);
    }};
}

Action Actions::AttackEnemy(int idx, int damage) {
    return {[=] (BattleContext &bc) {
        if (bc.monsters.arr[idx].isDeadOrEscaped()) {
            return;
        }

        bc.monsters.arr[idx].attacked(bc, damage);
        bc.checkCombat();
    }, false};
}

Action Actions::AttackAllEnemy(DamageMatrix damageMatrix) {
    return {[=] (BattleContext &bc) {
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            if (!bc.monsters.arr[i].isDeadOrEscaped()) {
                bc.monsters.arr[i].attacked(bc, damageMatrix[i]);
            }
        }

        bc.checkCombat();
    }, false};
}

Action Actions::DamageEnemy(int idx, int damage) {
    return {[=] (BattleContext &bc) {
        if (bc.monsters.arr[idx].isDeadOrEscaped()) {
            return;
        }

        bc.monsters.arr[idx].damage(bc, damage);
        bc.checkCombat();
    }, false};
}

Action Actions::DamageAllEnemy(int damage) { // todo this is probably broken
    return {[=] (BattleContext &bc) {
        for (int idx = 0; idx < bc.monsters.monsterCount; idx++) {
            if (!bc.monsters.arr[idx].isDeadOrEscaped()) {
                bc.monsters.arr[idx].damage(bc, damage); // possible should addToBot here todo
            }
        }
        bc.checkCombat();
    }, false};
}

Action Actions::AttackPlayer(int idx, int damage) {
    return {[=] (BattleContext &bc) {
        // NORMAL DamageAction cancels when its source dies or becomes halfDead.
        // In particular, Flame Barrier/Thorns can stop a multi-hit attack.
        if (bc.monsters.arr[idx].isDeadOrEscaped()) return;
        bc.player.attacked(bc, idx, damage);
    }, false};
}

Action Actions::RecordStolenGold(int idx, int goldAmount) {
    // Looter$1/$2 and Mugger$1/$2 do bookkeeping only. It is evaluated
    // after earlier queued effects, and is not the DAMAGE action below.
    return {[=] (BattleContext &bc) {
        const int amount = std::min(static_cast<int>(bc.player.gold), goldAmount);
        if (amount > 0) {
            if (bc.monsters.arr[idx].miscInfo > std::numeric_limits<int>::max() - amount)
                throw std::overflow_error("thief stolen-gold ledger exceeds native integer range");
            bc.monsters.arr[idx].miscInfo += amount;
            bc.setRequiresStolenGoldCheck(true);
        }
    }};
}

Action Actions::StealingAttackPlayer(int idx, int damage, int goldAmount) {
    return {[=] (BattleContext &bc) {
        // DamageAction cancels before stealing if its source is gone. Its
        // anonymous predecessor has no such source guard and can have run.
        if (bc.monsters.arr[idx].isDeadOrEscaped()) return;
        bc.player.gold -= std::min(static_cast<int>(bc.player.gold), goldAmount);
        bc.player.attacked(bc, idx, damage);
    }, false}; // DamageAction survives clearPostCombatActions.
}

Action Actions::DamagePlayer(int damage, bool selfDamage) {
    return {[=] (BattleContext &bc) {
        bc.player.damage(bc, damage, selfDamage);
    }, false};
}

Action Actions::VampireAttack(int damage) {
    return {[=] (BattleContext &bc) {
        const auto mIdx = 0;
        auto &m = bc.monsters.arr[mIdx]; // only used by shelled parasite so idx is 0
        bc.player.attacked(bc, mIdx, damage);
        if (m.isAlive()) {
            m.heal(std::min(damage, static_cast<int>(bc.player.lastAttackUnblockedDamage)));
        }
    }, false};
}

Action Actions::PlayerLoseHp(int hp, bool selfDamage) {
    return {[=] (BattleContext &bc) {
        bc.player.loseHp(bc, hp, selfDamage);
    }, false};
}

Action Actions::HealPlayer(int amount) {
    return {[=] (BattleContext &bc) {
        bc.player.heal(amount);
    }, false};
}

Action Actions::HealMonster(int idx, int amount) {
    return {[=] (BattleContext &bc) {
        bc.monsters.arr[idx].heal(amount);
    }, false}; // HealAction survives post-combat filtering.
}

Action Actions::MonsterGainBlock(int idx, int amount) {
    return {[=] (BattleContext &bc) {
        auto &m = bc.monsters.arr[idx];
        // GainBlockAction rejects dying/dead targets, not halfDead/escaping.
        if (!m.isDying() && !m.isDeadB) {
            m.addBlock(amount);
        }
    }, false};
}

Action Actions::SetMonsterMove(int idx, MonsterMoveId move) {
    return {[=] (BattleContext &bc) {
        // SetMoveAction has no dead/escaping guard and is not a DAMAGE action.
        bc.monsters.arr[idx].setMove(move);
    }};
}

Action Actions::RollMove(int monsterIdx) {
    return {[=] (BattleContext &bc) {
        Monster &m = bc.monsters.arr[monsterIdx];
        m.rollMove(bc);
    }};
}

Action Actions::HexaghostActivateOrb(int monsterIdx) {
    return {[=](BattleContext &bc) {
        auto &monster = bc.monsters.arr[monsterIdx];
        const int count = monster.miscInfo & Monster::HEXAGHOST_ORB_MASK;
        if (count >= 6) throw std::logic_error("Hexaghost cannot activate a seventh orb");
        monster.miscInfo = (monster.miscInfo & ~Monster::HEXAGHOST_ORB_MASK) | (count + 1);
        // Java ChangeState sets Inferno here, then the queued RollMove sets it
        // again. Both entries belong to the actual move history.
        if (count + 1 == 6) monster.setMove(MonsterMoveId::HEXAGHOST_INFERNO);
    }};
}

Action Actions::HexaghostDeactivateOrbs(int monsterIdx) {
    return {[=](BattleContext &bc) {
        bc.monsters.arr[monsterIdx].miscInfo &= ~Monster::HEXAGHOST_ORB_MASK;
    }};
}

Action Actions::BurnIncrease() {
    return {[](BattleContext &bc) {
        for (auto &card : bc.cards.discardPile) {
            if (card.getId() == CardId::BURN) card.upgrade();
        }
        for (auto &card : bc.cards.drawPile) {
            if (card.getId() == CardId::BURN) card.upgrade();
        }
        for (int i = 0; i < 3; ++i) {
            bc.cards.createTempCardInDiscard(CardInstance(CardId::BURN, true));
        }
    }};
}

Action Actions::ReactiveRollMove(int monsterIdx) {
    return {[=] (BattleContext &bc) {
        Monster &m = bc.monsters.arr[monsterIdx];
        // Preserve each RollMoveAction's position relative to block gain and
        // subsequent hits; do not coalesce all rolls into the first callback.
        m.rollMove(bc);
        m.setStatus<MS::REACTIVE>(m.getStatus<MS::REACTIVE>() - 1);
    }};
}

Action Actions::NoOpRollMove() {
    return {[=] (BattleContext &bc) {
        bc.noOpRollMove();
    }};
}

Action Actions::ChangeStance(Stance stance) {
    return sts::Action();
}

Action Actions::GainEnergy(int amount) {
    return {[=] (BattleContext &bc) {
        bc.player.gainEnergy(amount);
    }};
}

Action Actions::GainBlock(int amount) {
    return {[=] (BattleContext &bc) {
        bc.player.gainBlock(bc, amount);
    }, false};
}
//
//Action Actions::GainBlockFromCard(int amount) {
//    return {[=] (BattleContext &bc) {
//
//        // TODO
//        bc.player.block += amount;
//    };
//}

Action Actions::DrawCards(int amount) {
    return {[=] (BattleContext &bc) {
        bc.drawCards(amount);
    }};
}

Action Actions::ConditionalDraw(int amount, CardType restrictedType) {
    return {[=] (BattleContext &bc) {
        // On-play TOP reactions may change the hand before this action runs.
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            if (bc.cards.hand[i].getType() == restrictedType) {
                return;
            }
        }
        bc.addToTop(Actions::DrawCards(amount));
    }};
}

Action Actions::EmptyDeckShuffle() {
    return {[=] (BattleContext &bc) {
        java::Collections::shuffle(
                bc.cards.discardPile.begin(),
                bc.cards.discardPile.end(),
                java::Random(bc.shuffleRng.randomLong())
        );

        bc.cards.moveDiscardPileIntoToDrawPile();
        bc.applyPowersToHand(); // Soul completion refreshes the hand after shuffle.
    }};
}

Action Actions::ShuffleDrawPile() {
    return {[=] (BattleContext &bc) {
        bc.cards.knownDrawTop = 0;
        java::Collections::shuffle(
                bc.cards.drawPile.begin(),
                bc.cards.drawPile.end(),
                java::Random(bc.shuffleRng.randomLong())
        );
    }};
}

Action Actions::ShuffleTempCardIntoDrawPile(CardId id, int count) {
    return {[=] (BattleContext &bc) {
        CardInstance c(id);
        for (int i = 0; i < count; ++i) {
            const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
            bc.cards.createTempCardInDrawPile(idx, c);
        }
        if (count > 0) bc.applyPowersToHand();
    }};
}

Action Actions::PlayTopCard(int monsterTargetIdx, bool exhausts) {
    return {[=](BattleContext &bc) {
        bc.playTopCardInDrawPile(monsterTargetIdx, exhausts);
    }};
}

Action Actions::MayhemPlayTopCard() {
    return {[] (BattleContext &bc) {
        // MayhemPower$1 samples only when this wrapper executes, then appends
        // PlayTopCardAction behind the remaining start-of-turn actions. Do not
        // move the RNG call to power activation or play the card immediately.
        const int target = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
        bc.addToBot(Actions::PlayTopCard(target, false));
    }};
}


// todo fix the arguments are used wrongly all over ? what did this mean
// todo check for master reality
Action Actions::MakeTempCardInHand(CardId card, bool upgraded, int amount) {
    CardInstance c(card, upgraded);
    return Actions::MakeTempCardInHand(c, amount);
}

Action Actions::MakeTempCardInHand(CardInstance card, int amount) {
    // todo master reality when the action is created
    return {[=](BattleContext &bc) {
        for (int i = 0; i < amount; ++i) {
            bc.cards.createTempCardInHand(bc, card);
        }
    }};
}

Action Actions::MakeTempCardInDrawPile(const CardInstance &c, int amount, bool shuffleInto) {
    // the random calculation is done in an effect so it be wrong to do it here?
    return {[=](BattleContext &bc) {
        for (int i = 0; i < amount; ++i) {
            if (shuffleInto) {
                const int idx = bc.cards.drawPile.empty() ? 0 : bc.cardRandomRng.random(static_cast<int>(bc.cards.drawPile.size()-1));
                bc.cards.createTempCardInDrawPile(idx, c);
            }
            // todo else
        }
        if (amount > 0 && shuffleInto) bc.applyPowersToHand();
    }};
}

Action Actions::MakeTempCardInDiscard(const CardInstance &c, int amount) {
    CardInstance c_copy(c);
    return {[=](BattleContext &bc) { // todo maybe store by reference instead of copy if CardInstance is too big to fit directly in the lambda. Make a seperate space where the card will be stored
        for (int i = 0; i < amount; ++i) {
            bc.cards.createTempCardInDiscard(c_copy);
        }
        if (amount > 0) bc.applyPowersToHand();
    }};
}

Action Actions::MakeTempCardsInHand(std::vector<CardInstance> cards) {
    return {[=](BattleContext &bc) {
        for (auto c : cards) {
            bc.cards.createTempCardInHand(bc, c);
        }
    }};
}


Action Actions::DiscardNoTriggerCard() {
    return {[](BattleContext &bc) {
        const auto &c = bc.curCardQueueItem.card;
        bc.cards.notifyRemoveFromHand(c);
        bc.cards.moveToDiscardPile(c);
        bc.onCardDrawOrDiscard();
    }};
}

Action Actions::ClearCardQueue() {
    return {[](BattleContext &bc) {
        bc.cardQueue.clear();
    }};
}

Action Actions::DiscardAtEndOfTurn() {
    return {[] (BattleContext &bc) {
        bc.discardAtEndOfTurn();
    }};
}

Action Actions::DiscardAtEndOfTurnHelper() {
    return {[] (BattleContext &bc) {
        bc.discardAtEndOfTurnHelper();
    }};
}

Action Actions::RestoreRetainedCards(int count) {
    return {[=] (BattleContext &bc) {
        bc.restoreRetainedCards(count);
    }};
}

Action Actions::UnnamedEndOfTurnAction() {
    return {[] (BattleContext &bc) {
        // AbstractRoom$1 appends EndTurnAction. Existing queued callbacks
        // must still see turnHasEnded=false until that second action runs.
        bc.addToBot(Actions::EndTurnAction());
        // Java's intervening WaitAction has no simulation-state effect.
        if (!bc.skipMonsterTurn) {
            bc.addToBot(Actions::MonsterStartTurnAction());
        }
    }};
}

Action Actions::EndTurnAction() {
    return {[] (BattleContext &bc) {
        bc.turnHasEnded = true;
    }};
}

Action Actions::MonsterStartTurnAction() {
    return {[] (BattleContext &bc) {
        bc.monsterTurnIdx = 0;
        bc.monsters.applyPreTurnLogic(bc);
    }};
}

Action Actions::TriggerEndOfTurnOrbsAction() {
    return {[] (BattleContext &bc) {
        // todo
    }};
}

Action Actions::ExhaustTopCardInHand() {
    return {[=] (BattleContext &bc) {
        bc.exhaustTopCardInHand();
    }};
}

Action Actions::ExhaustSpecificCardInHand(int idx, std::int16_t uniqueId) {
    return {[=] (BattleContext &bc) {
        bc.exhaustSpecificCardInHand(idx, uniqueId);
    }};
}

Action Actions::DamageRandomEnemy(int damage) {
    return {[=] (BattleContext &bc) {
        const int idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
        if (idx == -1) {
            return;
        }
        bc.addToTop(Actions::DamageEnemy(idx, damage));
    }, false};
}

Action Actions::ExhaustRandomCardInHand(int count, bool anyNumber) {
    return {[=] (BattleContext &bc) {
        // ExhaustAction's mandatory whole-hand branch runs before isRandom.
        // Fiend Fire explicitly sets anyNumber=true and must still sample.
        const bool exhaustAll = !anyNumber && bc.cards.cardsInHand <= count;
        const int amount = exhaustAll ? bc.cards.cardsInHand : count;
        for (int i = 0; i < amount; ++i) {
            if (bc.cards.cardsInHand <= 0) {
                return;
            }
            const auto idx = exhaustAll ? bc.cards.cardsInHand - 1
                                       : bc.cards.getRandomCardIdxInHand(bc.cardRandomRng);
            auto c = bc.cards.hand[idx];
            bc.cards.removeFromHandAtIdx(idx);
            bc.triggerAndMoveToExhaustPile(c);
        }
    }};
}

Action Actions::MadnessAction() {
    return {[=] (BattleContext &bc) {

        bool haveNonZeroCost = false;
        bool haveNonZeroTurnCost = false;
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            const auto &c = bc.cards.hand[i];

            if (c.costForTurn > 0) {
                haveNonZeroTurnCost = true;
                break;
            }

            if (c.cost > 0) {
                haveNonZeroCost = true;
            }
        }

        const auto haveValidCard = haveNonZeroCost || haveNonZeroTurnCost;
        if (!haveValidCard) {
            return;
        }

        // always have 1 or more cards in hand here
//#pragma clang diagnostic push
//#pragma ide diagnostic ignored "EndlessLoop"
        while (true) {
            const auto randomIdx = bc.cardRandomRng.random(bc.cards.cardsInHand-1);
            auto &c = bc.cards.hand[randomIdx];

            if (haveNonZeroTurnCost) {
                if (c.costForTurn > 0) {
                    c.cost = 0;
                    c.costForTurn = 0;
                    break;
                } else {
                    continue;
                }

            } else {
                if (c.cost > 0) {
                    c.cost = 0;
                    c.costForTurn = 0;
                    break;

                } else {
                    continue;
                }
            }
        }
//#pragma clang diagnostic pop

    }};
}

Action Actions::RandomizeHandCost() {
    return {[=] (BattleContext &bc) {
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            auto &c = bc.cards.hand[i];
            if (c.cost >= 0) {
                int newCost = bc.cardRandomRng.random(3);
                // Java leaves an existing turn-only discount unchanged when
                // the sampled cost equals the card's current base cost.
                if (c.cost != newCost) {
                    c.cost = newCost;
                    c.costForTurn = newCost;
                }
            }
        }
    }};
}

Action Actions::GainBlockRandomEnemy(int sourceMonster, int amount) {
    return {[=] (BattleContext &bc) {
        int validIdxs[5];
        int validCount = 0;

        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto &m = bc.monsters.arr[i];
            if (i != sourceMonster && !m.isDying()) {
                validIdxs[validCount++] = i;
            }
        }

        int targetIdx;
        if (validCount > 0) {
            targetIdx = validIdxs[bc.aiRng.random(validCount - 1)];
        } else {
            targetIdx = sourceMonster;
        }

        bc.monsters.arr[targetIdx].addBlock(amount);
    }};
}

Action Actions::SummonGremlins() {
    // gremlin leader searches in the order 1, 2, 0 for open space

    return {[=] (BattleContext &bc) {
        int openIdxCount = 0;
        int newGremlinIdxs[2];
        if (bc.monsters.arr[1].isDying()) {
            newGremlinIdxs[openIdxCount++] = 1;
        }
        if (bc.monsters.arr[2].isDying()) {
            newGremlinIdxs[openIdxCount++] = 2;
        }
        if (openIdxCount < 2 && bc.monsters.arr[0].isDying()) {
            newGremlinIdxs[openIdxCount++] = 0;
        }
#ifdef sts_asserts
        assert(openIdxCount == 2);
#endif

        auto &gremlin0 = bc.monsters.arr[newGremlinIdxs[0]];
        auto &gremlin1 = bc.monsters.arr[newGremlinIdxs[1]];

        gremlin0 = Monster();
        gremlin1 = Monster();

        gremlin0.construct(bc, MonsterGroup::getGremlin(bc.aiRng), newGremlinIdxs[0]);
        gremlin1.construct(bc, MonsterGroup::getGremlin(bc.aiRng), newGremlinIdxs[1]);
        bc.monsters.monstersAlive += 2;

        if (bc.player.hasRelic<R::PHILOSOPHERS_STONE>()) {
            gremlin0.buff<MS::STRENGTH>(1);
            gremlin1.buff<MS::STRENGTH>(1);
        }
        gremlin0.buff<MS::MINION>();
        gremlin1.buff<MS::MINION>();

        gremlin0.rollMove(bc);
        gremlin1.rollMove(bc);
    }};
}

Action Actions::SpawnTorchHeads() {
    return {[=] (BattleContext &bc) {
        const auto spawnCount = 3-bc.monsters.monstersAlive;
#ifdef sts_asserts
        assert(spawnCount > 0);
#endif
        const int spawnIdxs[2] {(bc.monsters.arr[1].isDying() ? 1 : 0), 0};

        for (int i = 0; i < spawnCount; ++i) {
            const auto idx = spawnIdxs[i];
            auto &torchHead = bc.monsters.arr[idx];
            torchHead = Monster();
            torchHead.construct(bc, MonsterId::TORCH_HEAD, idx);
            torchHead.initHp(bc.monsterHpRng, bc.ascension); // bug somewhere in game
            torchHead.setMove(MMID::TORCH_HEAD_TACKLE);
            torchHead.buff<MS::MINION>();

            if (bc.player.hasRelic<R::PHILOSOPHERS_STONE>()) {
                torchHead.buff<MS::STRENGTH>(1);
            }
            ++bc.monsters.monstersAlive;
        }

        for (int i = 0; i < spawnCount; ++i) {
            bc.noOpRollMove();
        }
    }};
}

Action Actions::SpireShieldDebuff() {
    return {[] (BattleContext &bc) {
        if (bc.aiRng.randomBoolean()) {
            bc.player.debuff<PS::FOCUS>(-1);
        } else {
            bc.player.debuff<PS::STRENGTH>(-1);
        }
    }};
}


Action Actions::OnAfterCardUsed() {
    return {[] (BattleContext &bc) {
        bc.onAfterUseCard();
    }, false};
}

Action Actions::DiscoveryAction(CardType type, int amount) {
    return {[=] (BattleContext &bc) {
        bc.haveUsedDiscoveryAction = true;
        bc.openDiscoveryScreen(sts::generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, type), amount);
    }};
}

Action Actions::TransmutationAction(bool upgraded, int energy, bool useEnergy) {
    return {[=] (BattleContext &bc) {
        const auto effectAmount = energy + (bc.player.hasRelic<R::CHEMICAL_X>() ? 2 : 0);

        if (effectAmount == 0) {
            return;
        }

        std::vector<CardInstance> cards;
       for (int i = 0; i < effectAmount; ++i) {
           const auto cid = sts::getTrulyRandomColorlessCardInCombat(bc.cardRandomRng);
           CardInstance c(cid, upgraded);
           c.setCostForTurn(0);
           cards.push_back(c);
       }
       bc.addToBot( Actions::MakeTempCardsInHand(cards) );

       if (useEnergy) {
           bc.player.useEnergy(bc.player.energy);
       }
    }};
}

Action Actions::ViolenceAction(int count) { // todo a faster algorithm for inserting into the attack list
    return {[=] (BattleContext &bc) {
        // Generated cards can grow the draw pile beyond MAX_GROUP_SIZE.
        // Keep the insertion/shuffle order and RNG calls unchanged.
        std::vector<int> attackIdxList;
        attackIdxList.reserve(bc.cards.drawPile.size());
        for (int i = 0; i < bc.cards.drawPile.size(); ++i) {
            const auto &c = bc.cards.drawPile[i];
            if (c.getType() == CardType::ATTACK) {

                if (attackIdxList.empty()) {
                    attackIdxList.push_back(i);
                } else {
                    const auto randomIdx = bc.cardRandomRng.random(static_cast<int>(attackIdxList.size()) - 1);
                    attackIdxList.insert(attackIdxList.begin() + randomIdx, i);
                }
            }
        }

        if (attackIdxList.empty()) {
            return;
        }

        int removeIdxs[4];
        // hack to do this faster: the attackList is just pushed forward by i so we skip removing from bottom
        int i = 0;
        for (; i < count; ++i) {
            if (i >= static_cast<int>(attackIdxList.size())) {
                // Still remove the cards already transferred. Returning here
                // leaves the same instances in both draw and hand/discard.
                break;
            }

            java::Collections::shuffle(attackIdxList.begin()+i, attackIdxList.end(), java::Random(bc.shuffleRng.randomLong()));
            const auto removeIdx = attackIdxList[i];
            removeIdxs[i] = removeIdx;

            const auto &c = bc.cards.drawPile[removeIdx];
            if (bc.cards.cardsInHand == 10) {
                bc.cards.moveToDiscardPile(c);
                bc.onCardDrawOrDiscard();
            } else {
                bc.cards.moveToHand(c);
            }
        }

        std::sort(removeIdxs, removeIdxs+i);
        for (int x = i-1; x >= 0; --x) {
            const auto drawPileRemoveIdx = removeIdxs[x];
            bc.cards.removeFromDrawPileAtIdx(drawPileRemoveIdx);
        }
        bc.applyPowersToHand(); // DrawPileToHandAction / Violence, after actual moves.
    }};
}

Action Actions::BetterDiscardPileToHandAction(int amount, CardSelectTask task) {
    return {[=] (BattleContext &bc) {
        if (bc.cards.discardPile.empty() || amount <= 0) {
            return;
        }
        if (amount > CardSelectInfo::GRID_PICK_CAPACITY ||
            (amount != 1 && task != CardSelectTask::LIQUID_MEMORIES_POTION)) {
            throw std::invalid_argument("unsupported discard retrieval count/context");
        }
        if (bc.cards.discardPile.size() <= amount) {
            // Mandatory retrieval needs no UI when every card is selected.
            // A full hand leaves the remaining source cards and costs intact.
            const auto count = bc.cards.discardPile.size();
            for (int i = 0; i < count && bc.cards.cardsInHand < CardManager::MAX_HAND_SIZE; ++i)
                bc.chooseDiscardToHandCard(0, task == CardSelectTask::LIQUID_MEMORIES_POTION);
        } else {
            bc.openSimpleCardSelectScreen(task, amount);
        }
    }};
}

Action Actions::ArmamentsAction() {
    return {[=] (BattleContext &bc) {
        int canUpgradeCount = 0;
        int lastUpgradeIdx = 0;
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            if (bc.cards.hand[i].canUpgrade()) {
                ++canUpgradeCount;
                lastUpgradeIdx = i;
            }
        }

        if (canUpgradeCount == 0) {
            // do nothing

        } else if (canUpgradeCount == 1) {
            bc.cards.hand[lastUpgradeIdx].upgrade();

        } else {
            bc.openSimpleCardSelectScreen(CardSelectTask::ARMAMENTS, 1);
        }
    }};
}

Action Actions::DualWieldAction(int copyCount) {
    return {[=] (BattleContext &bc) {

//        fixed_list<CardInstance, 10> validCards;
//        fixed_list<CardInstance, 10> invalidCards;
//
//        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
//            const bool valid = bc.cards.hand[i].getType() == CardType::ATTACK || bc.cards.hand[i].getType() == CardType::POWER;
//            if (valid) {
//                validCards.push_back(bc.cards.hand[i]);
//
//            } else {
//                invalidCards.push_back(bc.cards.hand[i]);
//            }
//        }

        int validCount = 0;
        int lastValidIdx = 0;

        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            const bool valid = bc.cards.hand[i].getType() == CardType::ATTACK || bc.cards.hand[i].getType() == CardType::POWER;
            if (valid) {
                ++validCount;
                lastValidIdx = i;
            }
        }

        if (validCount == 0) {
            return;
        }

        if (validCount == 1) {
            for (int i = 0; i < copyCount; ++i) {
                auto c = bc.cards.hand[lastValidIdx].makeStatEquivalentCopy();
                bc.cards.createTempCardInHand(bc, c);
            }

        } else {
            bc.openSimpleCardSelectScreen(CardSelectTask::DUAL_WIELD, 1);
            bc.cardSelectInfo.dualWield_CopyCount() = copyCount;

        }


    }};
}

Action Actions::ExhumeAction() {
    return {[=] (BattleContext &bc) {
        if (bc.cards.exhaustPile.empty() || bc.cards.cardsInHand == 10) {
            return;
        }

        int nonExhumeCards = 0;
        int lastNonExhumeIdx = -1;
        for (int i = 0; i < bc.cards.exhaustPile.size(); ++i) {
            if (bc.cards.exhaustPile[i].id != CardId::EXHUME) {
                ++nonExhumeCards;
                lastNonExhumeIdx = i;
            }
        }

        if (nonExhumeCards == 0) {
            return;

        } else if (bc.cards.exhaustPile.size() == 1) {
            bc.chooseExhumeCard(lastNonExhumeIdx);

        } else {
            // Java checks the original pile size before temporarily removing
            // Exhumes. A larger pile still opens GRID with one eligible card.
            bc.openSimpleCardSelectScreen(CardSelectTask::EXHUME, 1);
        }

    }};
}

Action Actions::ForethoughtAction(bool upgraded) {
    return {[=] (BattleContext &bc) {
        if (bc.cards.cardsInHand == 0) {
            return;
        }

        if (!upgraded && bc.cards.cardsInHand == 1) {
            bc.chooseForethoughtCard(0);
        } else {
            bc.openSimpleCardSelectScreen(CardSelectTask::FORETHOUGHT,
                                          upgraded ? bc.cards.cardsInHand : 1);
            bc.cardSelectInfo.canPickAnyNumber = upgraded;
            bc.cardSelectInfo.canPickZero = upgraded;
        }
    }};
}

Action Actions::HeadbuttAction() {
    return {[=] (BattleContext &bc) {
        if (bc.cards.discardPile.empty()) {
            return;

        } else if (bc.cards.discardPile.size() == 1) {
            bc.chooseHeadbuttCard(0);
        } else {
            bc.openSimpleCardSelectScreen(CardSelectTask::HEADBUTT, 1);
        }
    }};
}

Action Actions::ChooseExhaustOne() {
    return {[=] (BattleContext &bc) {
        if (bc.cards.cardsInHand == 0) {
            return;

        } else if (bc.cards.cardsInHand == 1) {
            bc.chooseExhaustOneCard(0);

        } else {
            bc.openSimpleCardSelectScreen(CardSelectTask::EXHAUST_ONE, 1);

        }
    }};
}

Action Actions::DrawToHandAction(CardSelectTask task, CardType cardType) {
    return {[=] (BattleContext &bc) {
        int count = 0;
        int idx = 0;

        for (int i = 0; i < bc.cards.drawPile.size(); ++i) {
            const auto &c = bc.cards.drawPile[i];
            if (c.getType() == cardType) {
                if (count > 0) {
                    // for keeping rng consistent with game
                    // the game creates a temporary list with the skills
                    bc.cardRandomRng.random(count - 1);
                }
                idx = i;
                ++count;
            }
        }

        if (count == 0) {
            return;
        }

        if (count == 1) {
            bc.chooseDrawToHandCards(&idx, 1);

        } else {
            bc.openSimpleCardSelectScreen(task, 1);
        }

    }};
}

Action Actions::WarcryAction() {
    return {[=] (BattleContext &bc) {
        // todo if the handSize equals or less than the cardsToChoose just choose them here
        if (bc.cards.cardsInHand == 0) {
            return;
        }

        if (bc.cards.cardsInHand == 1) {
            bc.cardRandomRng.random(1);
            bc.chooseWarcryCard(0);

        } else {
            bc.openSimpleCardSelectScreen(CardSelectTask::WARCRY, 1);
        }
    }};
}

Action Actions::ResolveUnplayedCard(const CardQueueItem &x) {
    return {[=] (BattleContext &bc) {
        auto item = x;
        // Each failed/cancelled autoplay resolves its own UseCardAction. The last
        // card processed may have a different persistent exhaust attribute.
        item.exhaustOnUse |= item.card.doesExhaust();
        item.triggerOnUse = false;
        bc.curCardQueueItem = item;
        bc.onAfterUseCard();

    }, false};
}

Action Actions::UpgradeAllCardsInHand() {
    return {[=] (BattleContext &bc) {
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            if (bc.cards.hand[i].canUpgrade()) {
                bc.cards.hand[i].upgrade();
            }
        }
    }};
}

Action Actions::EssenceOfDarkness(int darkOrbsPerSlot) {
    return sts::Action();
}

Action Actions::IncreaseOrbSlots(int count) {
    return sts::Action();
}

Action Actions::SuicideAction(int monsterIdx, bool triggerRelics) {
    return {[=] (BattleContext &bc) {
        // Java sets HP to zero and calls die; block, Intangible and damage
        // callbacks must not prevent a suicide. It does not filter the queue.
        bc.monsters.arr[monsterIdx].suicideAction(bc, triggerRelics);
    }, false};
}

Action Actions::EscapeAction(int monsterIdx) {
    return {[=] (BattleContext &bc) {
        bc.monsters.arr[monsterIdx].escape(bc);
    }};
}

Action Actions::PoisonLoseHpAction(int target, int amount) {
    return {[target, amount](BattleContext &bc) {
        auto &monster = bc.monsters.arr[target];
        // HP_LOSS bypasses block and NORMAL-attack callbacks. The amount was
        // captured by PoisonPower atStartOfTurn, before earlier deaths can
        // transfer more poison to this target through The Specimen.
        if (monster.curHp > 0 && !monster.isDying() && !monster.isEscaping()) {
            const int damage = monster.hasStatus<MS::INTANGIBLE>()
                ? std::min(amount, 1) : amount;
            monster.damageUnblockedHelper(bc, std::max(0, damage));
        }
        // Subclass death may already have removed Poison (Darkling/Awakened).
        if (monster.hasStatus<MS::POISON>()) monster.decrementStatus<MS::POISON>();
        bc.checkCombat();
    }, false}; // Java PoisonLoseHpAction is DAMAGE, not a discardable callback.
}

Action Actions::RemovePlayerDebuffs() {
    return {[] (BattleContext &bc) {
        bc.player.removeDebuffs();
        bc.applyPowersToHand();
    }};
}

Action Actions::UpgradeRandomCardAction() {
    return {[] (BattleContext &bc) {
        fixed_list<int,10> upgradeableHandIdxs;
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            if (bc.cards.hand[i].canUpgrade()) {
                upgradeableHandIdxs.push_back(i);
            }
        }

        if (upgradeableHandIdxs.empty()) {
            return;
        }

        java::Collections::shuffle(
                upgradeableHandIdxs.begin(),
                upgradeableHandIdxs.end(),
                java::Random(bc.shuffleRng.randomLong())
        );

        const auto upgradeIdx = upgradeableHandIdxs[0];
        bc.cards.hand[upgradeIdx].upgrade();
    }};
}

Action Actions::CodexAction() {
    return {[] (BattleContext &bc) {
        if (bc.monsters.areMonstersBasicallyDead()) return;
        bc.openSimpleCardSelectScreen(CardSelectTask::CODEX, 1);
        bc.cardSelectInfo.codexCards() =
                generateDiscoveryCards(bc.cardRandomRng, CharacterClass::IRONCLAD, CardType::INVALID);
    }};
}

Action Actions::ExhaustMany(int limit) {
    return {[=] (BattleContext &bc) {
        if (bc.cards.cardsInHand == 0) return;
        bc.openSimpleCardSelectScreen(CardSelectTask::EXHAUST_MANY, limit);
        bc.cardSelectInfo.canPickZero = true;
        bc.cardSelectInfo.canPickAnyNumber = true;
    }};
}

Action Actions::GambleAction() {
    return {[] (BattleContext &bc) {
        bc.openSimpleCardSelectScreen(CardSelectTask::GAMBLE, bc.cards.cardsInHand);
        bc.cardSelectInfo.canPickZero = true;
        bc.cardSelectInfo.canPickAnyNumber = true;
    }};
}

Action Actions::ToolboxAction() {
    return {[] (BattleContext &bc) {
        bc.openSimpleCardSelectScreen(CardSelectTask::DISCOVERY, 1);
        bc.cardSelectInfo.discovery_CopyCount() = 1;
        bc.cardSelectInfo.discoveryFreeForTurn = false;
        bc.cardSelectInfo.discovery_Cards() =
                generateDiscoveryCards(bc.cardRandomRng, bc.player.cc, CardType::STATUS); // status is mapped to colorless
    }};
}

Action Actions::DualityAction() {
    return {[](BattleContext &bc) {
        bc.player.buff<PS::DEXTERITY>(1);
        bc.player.debuff<PS::LOSE_DEXTERITY>(1);
        bc.applyPowersToHand();
    }};
}

Action Actions::ApotheosisAction() {
    return {[](BattleContext &bc) {

        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            auto &c = bc.cards.hand[i];
            if (c.canUpgrade()) {
                c.upgrade();
            }
        }

        for (auto &c : bc.cards.drawPile) {
            if (c.canUpgrade()) {
                c.upgrade();
            }
        }

        for (auto &c : bc.cards.discardPile) {
            if (c.canUpgrade()) {
                c.upgrade();
            }
        }

        for (auto &c : bc.cards.exhaustPile) {
            if (c.canUpgrade()) {
                c.upgrade();
            }
        }
    }};
}

Action Actions::DropkickAction(int targetIdx, int damage) {
    return {[=](BattleContext &bc) {
        // Java tests the target's power at update, but DamageInfo was captured
        // by the card at use. A non-null dead target can still have Vulnerable.
        if (bc.monsters.arr[targetIdx].hasStatus<MS::VULNERABLE>()) {
            bc.addToTop(Actions::DrawCards(1));
            bc.addToTop(Actions::GainEnergy(1));
        }

        bc.addToTop(Actions::AttackEnemy(targetIdx, damage));
    }};
}

Action Actions::EnlightenmentAction(bool upgraded) {
    return {[=] (BattleContext &bc) {
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            auto &c = bc.cards.hand[i];
            if (c.costForTurn > 1) {
                c.costForTurn = 1;
            }
            if (upgraded && c.cost > 1) {
                // Base and turn costs are reduced independently in Java;
                // a card already at zero for this turn must stay at zero.
                c.cost = 1;
            }
        }
    }};
}

Action Actions::EntrenchAction() {
    return {[=] (BattleContext &bc) {
        bc.player.gainBlock(bc, bc.player.block);
    }};
}

Action Actions::FeedAction(int idx, int damage, bool upgraded) {
    return {[=] (BattleContext &bc) {
        auto &m = bc.monsters.arr[idx];
        if (m.isDeadOrEscaped()) {
            return;
        }
        bc.monsters.arr[idx].attacked(bc, damage);

        const bool effectTriggered = !m.hasStatus<MS::MINION>()
                && !m.isAlive()
                && !m.isHalfDead()
                && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);

        if (effectTriggered) {
            bc.player.increaseMaxHp(upgraded ? 4 : 3);
        }

        bc.checkCombat();
    }, false};
}

Action Actions::FiendFireAction(int targetIdx, int calculatedDamage) {
    return {[=] (BattleContext &bc) {
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            bc.addToTop(Actions::AttackEnemy(targetIdx, calculatedDamage));
        }

        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            bc.addToTop( Actions::ExhaustRandomCardInHand(1, true) );
        }
    }};
}

Action Actions::HandOfGreedAction(int idx, int damage, bool upgraded) {
    return {[=] (BattleContext &bc) {
        auto &m = bc.monsters.arr[idx];
        if (m.isDeadOrEscaped()) {
            return;
        }
        bc.monsters.arr[idx].attacked(bc, damage);

        const bool effectTriggered = !m.hasStatus<MS::MINION>()
                && !m.isAlive()
                && !m.isHalfDead()
                && !(m.hasStatus<MS::REGROW>() && bc.monsters.monstersAlive > 0);

        if (effectTriggered) {
            bc.player.gainGold(bc, upgraded ? 25 : 20);
        }

        bc.checkCombat();
    }, false};
}

Action Actions::LimitBreakAction() {
    return {[] (BattleContext &bc) {
        const int strength = bc.player.getStatus<PS::STRENGTH>();
        // Java captures the current amount here, then prepends ApplyPower.
        // Negative Strength is a DEBUFF and can consume/be blocked by Artifact.
        if (strength > 0) {
            bc.addToTop(Actions::BuffPlayer<PS::STRENGTH>(strength));
        } else if (strength < 0) {
            bc.addToTop(Actions::DebuffPlayer<PS::STRENGTH>(strength, false));
        }
    }};
}

Action Actions::ReaperAction(DamageMatrix damageMatrix) {
    return {[=] (BattleContext &bc) {

        int healAmount = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            auto &m = bc.monsters.arr[i];
            if (m.isDeadOrEscaped()) {
                continue;
            }
            int preDamageHp = m.curHp;
            m.attacked(bc, damageMatrix[i]);
            healAmount += preDamageHp-m.curHp;
        }

        if (healAmount > 0) {
            bc.addToBot( Actions::HealPlayer(healAmount) );
        }

        // The heal is enqueued before victory filtering and must survive it.
        bc.checkCombat();
    }, false};
}

Action Actions::IncreaseCardSpecialData(std::int16_t uniqueId, int amount) {
    return {[=] (BattleContext &bc) {
        bc.increaseCardSpecialData(uniqueId, amount);
    }};
}

Action Actions::RitualDaggerAction(int idx, int damage, std::int16_t uniqueId, int increaseAmount) {
    return {[=] (BattleContext &bc) {
        auto &m = bc.monsters.arr[idx];
        if (m.isDeadOrEscaped()) {
            return;
        }
        bc.monsters.arr[idx].attacked(bc, damage);

        const bool shouldUpgrade = !m.hasStatus<MS::MINION>()
                                   && !m.isAlive()
                                   && !m.isHalfDead();
        if (shouldUpgrade) {
            bc.increaseCardSpecialData(uniqueId, increaseAmount);
        }

        bc.checkCombat();
    }, false};
}

Action Actions::SecondWindAction(int blockPerCard) {
    return {[=] (BattleContext &bc) {
        int cardIdxsToExhaust[10];
        int toExhaustCount = 0;

        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            const auto &c = bc.cards.hand[i];
            if (c.getType() != CardType::ATTACK) {
                cardIdxsToExhaust[toExhaustCount++] = i;
                bc.addToTop( Actions::GainBlock(blockPerCard) );
            }
        }

        for (int i = 0; i < toExhaustCount; ++i) {
            const auto handIdx = cardIdxsToExhaust[i];
            const auto &c = bc.cards.hand[handIdx];

            bc.addToTop( Actions::ExhaustSpecificCardInHand(handIdx, c.uniqueId) );
        }
    }};
}

Action Actions::SeverSoulExhaustAction() {
    return {[=] (BattleContext &bc) {
        // ExhaustAllNonAttackAction visits the original hand in order and
        // prepends each exhaust, so the last eligible card resolves first.
        // These must run before the already-queued attack/UseCardAction:
        // Charon's Ashes can kill its target, and lethal damage can clear
        // pending exhaust actions. Capture physical IDs, not a moving index.
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            const auto &c = bc.cards.hand[i];
            if (c.getType() != CardType::ATTACK) {
                bc.addToTop( Actions::ExhaustSpecificCardInHand(i, c.getUniqueId()) );
            }
        }
    }};
}

Action Actions::SwordBoomerangAction(const CardInstance &card) {
    return {[=](BattleContext &bc) {
        const auto idx = bc.monsters.getRandomMonsterIdx(bc.cardRandomRng, true);
        if (idx == -1) {
            return;
        }

        // Unlike a captured multiDamage array, this Java action calculates
        // against its randomly chosen target on every hit, using the real card.
        const int damage = bc.calculateCardDamage(card, idx, 3);
        bc.addToTop(Actions::AttackEnemy(idx, damage));
    }};
}

Action Actions::SpotWeaknessAction(int target, int strength) {
    return {[=](BattleContext &bc) {
        if (bc.monsters.arr[target].isAttacking()) {
            bc.player.buff<PS::STRENGTH>(strength);
            bc.applyPowersToHand();
        }
    }};
}

Action Actions::WhirlwindAction(DamageMatrix damageMatrix, int energy, bool useEnergy) {
    return {[=] (BattleContext &bc) {
        const auto effectAmount = energy + (bc.player.hasRelic<R::CHEMICAL_X>() ? 2 : 0);
        if (effectAmount > 0) {
            // The Java action appends all hits after already-pending callbacks
            // (including UseCardAction), rather than dealing the first hit now.
            bc.addToBot(Actions::AttackAllMonsterRecursive(damageMatrix, effectAmount));
            if (useEnergy) bc.player.useEnergy(bc.player.energy);
        }
    }};
}

Action Actions::AttackAllMonsterRecursive(DamageMatrix matrix, int timesRemaining) {
    return {[=] (BattleContext &bc) {
        if (timesRemaining <= 0) {
            return;
        }

        if (timesRemaining > 1) {
            // Reserve the next hit before dispatching this hit's addToTop
            // reactions, so retaliation can kill the player between waves.
            bc.addToTop(AttackAllMonsterRecursive(matrix, timesRemaining-1));
        }
        Actions::AttackAllEnemy(matrix).actFunc(bc);
    }, false};
}
