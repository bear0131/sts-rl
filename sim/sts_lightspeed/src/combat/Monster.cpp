//
// Created by gamerpuppy on 7/5/2021.
//

#include <algorithm>

#include "combat/Monster.h"

#include "game/Random.h"
#include "combat/BattleContext.h"
#include "game/GameContext.h"

using namespace sts;

void Monster::setRandomHp(Random &hpRng, bool higherHp) {
    auto hpRange = monsterHpRange[static_cast<int>(id)][higherHp ? 1 : 0];
    curHp = hpRng.random(hpRange[0], hpRange[1]);
    maxHp = curHp;
}

void Monster::applyStartOfTurnPowers(BattleContext &bc) {
    if (!hasStatus<MS::BARRICADE>()) {
        block = 0;
    }

    if (hasStatus<MS::CHOKED>()) {
        setHasStatus<MS::CHOKED>(false);
    }

    if (hasStatus<MS::FLIGHT>()) {
        setStatus<MS::FLIGHT>(bc.ascension >= 17 ? 4 : 3);
    }

    if (hasStatus<MS::INVINCIBLE>()) {
        setStatus<MS::INVINCIBLE>(bc.ascension >= 19 ? 200 : 300);
    }

    if (hasStatus<MS::POISON>() && !bc.monsters.areMonstersBasicallyDead()) {
        bc.addToBot(Actions::PoisonLoseHpAction(idx, getStatus<MS::POISON>()));
    }
}

void Monster::applyEndOfTurnTriggers(BattleContext &bc) {
    // Monster Powers atEndOfTurnPreEndTurnCards and atEndOfTurn
    if (hasStatus<MS::METALLICIZE>()) {
        addBlock(getStatus<MS::METALLICIZE>());
    }

    if (hasStatus<MS::MALLEABLE>()) {
        setStatus<MS::MALLEABLE>(3);
    }

    if (hasStatus<MS::PLATED_ARMOR>()) {
        addBlock(getStatus<MS::PLATED_ARMOR>());
    }

    if (hasStatus<MS::INTANGIBLE>()) {
        decrementStatus<MS::INTANGIBLE>();
    }

    if (hasStatus<MS::REGEN>() && !halfDead && !isDying() && !isDeadB) {
        bc.addToBot(Actions::HealMonster(idx, getStatus<MS::REGEN>()));
    }

    if (hasStatus<MS::SHACKLED>()) {
        buff<MS::STRENGTH>(getStatus<MS::SHACKLED>());
        removeStatus<MS::SHACKLED>();
    }
}

void Monster::applyEndOfRoundPowers(BattleContext &bc) {
    // Monster Powers atEndOfRound
    if (hasStatus<MS::RITUAL>()) { // sort of a hack, RitualPower contains a skipFirst boolean
        if (wasJustApplied<MS::RITUAL>()) {
            setJustApplied<MS::RITUAL>(false);
        } else {
            buff<MS::STRENGTH>(getStatus<MS::RITUAL>());
        }
    }

    if (hasStatus<MS::SLOW>()) {
        setStatus<MS::SLOW>(0);
    }

    if (hasStatus<MS::LOCK_ON>()) {
        decrementStatus<MS::LOCK_ON>();
    }

    if (hasStatus<MS::WEAK>()) {
        if (wasJustApplied<MS::WEAK>()) {
            setJustApplied<MS::WEAK>(false);
        } else {
            decrementStatus<MS::WEAK>();
        }
    }

    if (hasStatus<MS::VULNERABLE>()) {
        if (wasJustApplied<MS::VULNERABLE>()) {
            setJustApplied<MS::VULNERABLE>(false);
        } else {
            decrementStatus<MS::VULNERABLE>();
        }
    }

    if (hasStatus<MS::GENERIC_STRENGTH_UP>()) {
        buff<MS::STRENGTH>(getStatus<MS::GENERIC_STRENGTH_UP>());
    }

}

void Monster::construct(BattleContext &bc, MonsterId monsterId, int monsterIdx) {
    *this = Monster();
    this->id = monsterId;
    this->idx = monsterIdx;
    isDyingB = false;
    initHp(bc.monsterHpRng, bc.ascension);

    switch (id) {
        case MonsterId::LOOTER:
        case MonsterId::MUGGER:
            thiefGoldAmount = bc.ascension >= 17 ? 20 : 15;
            break;

        case MonsterId::GREEN_LOUSE:
        case MonsterId::RED_LOUSE:
            if (bc.ascension >= 2) {
                miscInfo = bc.monsterHpRng.random(6, 8);
            } else {
                miscInfo = bc.monsterHpRng.random(5, 7);
            }
            break;

        case MonsterId::DARKLING:
            if (bc.ascension >= 2) {
                miscInfo = bc.monsterHpRng.random(9, 13);
            } else {
                miscInfo = bc.monsterHpRng.random(7, 11);
            }
            break;

        default:
            break;
    }
}

const char *Monster::getName() const {
    return monsterIdStrings[static_cast<int>(id)];
}

bool Monster::hasStatusInternal(MonsterStatus s) const {
    return statusBits & (1ULL << (int)s);
}

int Monster::getStatusInternal(MonsterStatus s) const {
    if (s == MS::STRENGTH) {
        return strength;
    }

    if (!hasStatusInternal(s)) {
        return 0;
    }

    switch (s) {
        case MS::ARTIFACT:
            return artifact;

        case MS::BLOCK_RETURN:
            return blockReturn;

        case MS::CHOKED:
            return choked;

        case MS::CORPSE_EXPLOSION:
            return corpseExplosion;

        case MS::LOCK_ON:
            return lockOn;

        case MS::MARK:
            return mark;

        case MS::METALLICIZE:
            return metallicize;

        case MS::PLATED_ARMOR:
            return platedArmor;

        case MS::POISON:
            return poison;

        case MS::REGEN:
            return regen;

        case MS::SHACKLED:
            return shackled;

        case MS::VULNERABLE:
            return vulnerable;

        case MS::WEAK:
            return weak;

        case MS::ANGRY:
        case MS::BEAT_OF_DEATH:
        case MS::CURIOSITY:
        case MS::CURL_UP:
        case MS::ENRAGE:
        case MS::FADING:
        case MS::FLIGHT:
        case MS::GENERIC_STRENGTH_UP:
        case MS::INTANGIBLE:
        case MS::MALLEABLE:
        case MS::MODE_SHIFT:
        case MS::RITUAL:
        case MS::SLOW:
        case MS::SPORE_CLOUD:
        case MS::THIEVERY:
        case MS::THORNS:
        case MS::TIME_WARP:
            return uniquePower0;

        case MS::INVINCIBLE:
        case MS::REACTIVE:
        case MS::SHARP_HIDE:
            return uniquePower1;

        // boolean powers
        case MS::ASLEEP:
        case MS::BARRICADE:
        case MS::MINION:
        case MS::MINION_LEADER:
        case MS::PAINFUL_STABS:
        case MS::REGROW:
        case MS::SHIFTING:
        case MS::STASIS:
            return true; // already did status check above

        default:
#ifdef sts_asserts
            assert(false);
#endif
            return 0;
    }
}

bool Monster::isAlive() const {
    return curHp > 0;
}

bool Monster::isTargetable() const {
    return id != MonsterId::INVALID && !isDeadOrEscaped();
}

bool Monster::isDying() const {
    return isDyingB;
}

bool Monster::isEscaping() const {
    return isEscapingB;
}

bool Monster::isDeadOrEscaped() const {
    return isDying() || isHalfDead() || isEscaping();
}

bool Monster::isHalfDead() const {
    return halfDead;
}

bool Monster::doesEscapeNext() const {
    return escapeNext;
}

bool Monster::isAttacking() const {
    return isMoveAttack(moveHistory[0]);
}

void Monster::heal(int amount) {
#ifdef sts_asserts
    if (amount < 0) {
        assert (false);
    }
#endif

    if (!isDying()) curHp = std::min(maxHp, curHp + amount);
}

void Monster::addBlock(int amount) {
    block = std::clamp(block, 0, MAX_BLOCK);
    if (amount > 0) {
        block += std::min(amount, MAX_BLOCK - block);
    }
}

void Monster::triggerDeathEffects(BattleContext &bc) {
    const bool battleEnding = bc.monsters.areMonstersBasicallyDead();
    if (curHp <= 0 && hasStatus<MS::SPORE_CLOUD>() && !battleEnding) {
        bc.addToTop(Actions::DebuffPlayer<PS::VULNERABLE>(getStatus<MS::SPORE_CLOUD>(), true));
    }

    if (curHp <= 0 && hasStatus<MS::STASIS>()) {
        returnStasisCard(bc);
    }

    if (curHp <= 0 && hasStatus<MS::CORPSE_EXPLOSION>() && !battleEnding) {
        int damage = maxHp * getStatus<MS::CORPSE_EXPLOSION>();
        bc.addToBot( Actions::DamageAllEnemy(damage) );
    }

    if (curHp == 0 && bc.player.hasRelic<RelicId::GREMLIN_HORN>() && !battleEnding) {
        bc.addToBot( Actions::GainEnergy(1) );
        bc.addToBot( Actions::DrawCards(1) );
    }

    if (bc.player.hasRelic<RelicId::THE_SPECIMEN>() && hasStatus<MS::POISON>() && !battleEnding) {
        const int poisonAmount = getStatus<MS::POISON>();
        bc.addToBot({[poisonAmount](BattleContext &ctx) {
            const int target = ctx.monsters.getRandomMonsterIdx(ctx.cardRandomRng, true);
            if (target >= 0) ctx.addToTop(Actions::DebuffEnemy<MS::POISON>(target, poisonAmount, false));
        }});
    }

}

void Monster::finishDeath(BattleContext &bc, bool triggerRelics, bool removeFromLivingCount) {
    // AbstractMonster.die(boolean), also called directly by SuicideAction.
    // It deliberately bypasses subclass damage/revival logic.
    if (isDying()) return;
    inactiveIntentDamage = getIntentDamage(bc, getMoveBaseDamage(bc).damage);
    isDyingB = true;
    if (removeFromLivingCount && !isEscaping()) --bc.monsters.monstersAlive;
    if (triggerRelics) triggerDeathEffects(bc);
    if (bc.monsters.areMonstersBasicallyDead() && bc.outcome == Outcome::UNDECIDED) {
        bc.outcome = Outcome::PLAYER_VICTORY;
    }
}

void Monster::die(BattleContext &bc) {
    if (isDying() || isHalfDead()) return;
    if (id == MonsterId::MUGGER) {
        bc.aiRng.random(2); // Mugger.playDeathSfx, unlike Looter's MathUtils RNG.
    }
    if (id == MonsterId::AWAKENED_ONE && !miscInfo) {
        if (!isEscaping()) --bc.monsters.monstersAlive;
        halfDead = true;
        triggerDeathEffects(bc);
        bc.addToTop(Actions::ClearCardQueue());
        removeDebuffs();
        removeStatus<MS::CURIOSITY>();
        setMove(MonsterMoveId::AWAKENED_ONE_REBIRTH);
        const int target = idx;
        bc.addToBot({[target](BattleContext &ctx) {
            ctx.monsters.arr[target].setMove(MMID::AWAKENED_ONE_REBIRTH);
        }});
        return;
    }

    if (id == MonsterId::DARKLING) {
        if (!isEscaping()) --bc.monsters.monstersAlive;
        halfDead = true;
        triggerDeathEffects(bc);
        resetAllStatusEffects();
        bool allDown = true;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto &m = bc.monsters.arr[i];
            if (m.id == MonsterId::DARKLING && !m.isHalfDead()) allDown = false;
        }
        if (allDown) {
            // Java clears the last one's halfDead flag, then calls die on the
            // whole group. The Darklings' living-count decrements already
            // happened; other entities (in custom encounters) still count.
            halfDead = false;
            for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                auto &other = bc.monsters.arr[i];
                other.finishDeath(bc, true,
                    other.id != MonsterId::DARKLING && other.isTargetable());
            }
        } else if (moveHistory[0] != MMID::DARKLING_REGROW) {
            setMove(MMID::DARKLING_REGROW);
            const int target = idx;
            bc.addToBot({[target](BattleContext &ctx) {
                ctx.monsters.arr[target].setMove(MMID::DARKLING_REGROW);
            }});
        }
        return;
    }

    finishDeath(bc, true, true);

    // These callbacks enqueue effects after the current damage action. In
    // particular, an AOE still hits the remaining slots before these execute.
    if (id == MonsterId::BRONZE_AUTOMATON || id == MonsterId::THE_COLLECTOR ||
        id == MonsterId::REPTOMANCER) {
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            if (!bc.monsters.arr[i].isDying()) {
                bc.addToTop(Actions::SuicideAction(i, true));
            }
        }
    } else if (id == MonsterId::GREMLIN_LEADER ||
               (id == MonsterId::AWAKENED_ONE && miscInfo)) {
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const auto &other = bc.monsters.arr[i];
            if (!other.isDying() &&
                (id == MonsterId::GREMLIN_LEADER || other.id == MonsterId::CULTIST)) {
                bc.addToBot(Actions::EscapeAction(i));
            }
        }
    }

    if (id == MonsterId::AWAKENED_ONE) {
        // Its damage override executes this block even after final-life die().
        // Keep that second callback pass; do not normalize it away as redundant.
        triggerDeathEffects(bc);
        bc.addToTop(Actions::ClearCardQueue());
        removeDebuffs();
        removeStatus<MS::CURIOSITY>();
        setMove(MMID::AWAKENED_ONE_REBIRTH);
        const int target = idx;
        bc.addToBot({[target](BattleContext &ctx) {
            ctx.monsters.arr[target].setMove(MMID::AWAKENED_ONE_REBIRTH);
        }});
    }
}

void Monster::suicideAction(BattleContext &bc, bool triggerRelics) {
    const bool wasCounted = isTargetable();
    curHp = 0;
    finishDeath(bc, triggerRelics, wasCounted);
}

void Monster::escape(BattleContext &bc) {
    // A previously queued escape can execute after another action killed its
    // target. Mark escaping, but never decrement the living count twice.
    const bool wasTargetable = isTargetable();
    inactiveIntentDamage = getIntentDamage(bc, getMoveBaseDamage(bc).damage);
    isEscapingB = true;
    if (wasTargetable) --bc.monsters.monstersAlive;
    if (bc.monsters.areMonstersBasicallyDead() && bc.outcome == Outcome::UNDECIDED) {
        bc.outcome = Outcome::PLAYER_VICTORY;
    }
}

void Monster::attackedUnblockedHelper(BattleContext &bc, int damage) {
    // AbstractMonster.damage dispatches onAttacked even for a fully blocked
    // hit. Resolve damage modifiers before HP-loss-dependent callbacks.
    if (bc.player.hasRelic<RelicId::THE_BOOT>() && damage > 0 && damage < 5) {
        damage = 5;
    }

    if (hasStatus<MS::INVINCIBLE>()) {
        damage = std::min(damage, getStatus<MS::INVINCIBLE>());
        setStatus<MS::INVINCIBLE>(getStatus<MS::INVINCIBLE>() - damage);
    }

    if (damage > 0 && bc.player.hasStatus<PS::ENVENOM>()) {
        bc.addToTop(Actions::DebuffEnemy<MS::POISON>(idx, bc.player.getStatus<PS::ENVENOM>()));
    }

    if (damage > 0 && hasStatus<MS::ANGRY>()) {
        const int target = idx;
        const int strength = getStatus<MS::ANGRY>();
        bc.addToTop({[target, strength](BattleContext &ctx) {
            auto &monster = ctx.monsters.arr[target];
            if (!monster.isDeadOrEscaped()) monster.buff<MS::STRENGTH>(strength);
        }});
    }

    // Thorns does not require HP loss; it is not an alternative to a damage
    // modifier. A fatal hit still schedules its retaliation.
    if (hasStatus<MS::THORNS>()) {
        bc.addToTop(Actions::DamagePlayer(getStatus<MS::THORNS>()));
    }
    if (damage <= 0) return;

    if (hasStatus<MS::PLATED_ARMOR>()) {
        const int target = idx;
        bc.addToBot({[target](BattleContext &ctx) {
            auto &monster = ctx.monsters.arr[target];
            if (!monster.hasStatus<MS::PLATED_ARMOR>()) return;
            monster.decrementStatus<MS::PLATED_ARMOR>();
            if (!monster.hasStatus<MS::PLATED_ARMOR>() && monster.id == MonsterId::SHELLED_PARASITE) {
                // ReducePower/RemoveSpecificPower enqueue ChangeState at bottom.
                ctx.addToBot({[target](BattleContext &next) {
                    next.monsters.arr[target].setMove(MMID::SHELLED_PARASITE_STUNNED);
                }});
            }
        }});

    } else if (hasStatus<MS::CURL_UP>() && damage < curHp) {
        bc.addToBot(Actions::MonsterGainBlock(this->idx, getStatus<MS::CURL_UP>()) );
        setHasStatus<MS::CURL_UP>(false);

    } else if (hasStatus<MS::FLIGHT>() && static_cast<float>(damage) / 2.0f < curHp) {
        // FlightPower reuses its final-receive calculation for this guard,
        // even though damage has already passed through that calculation.
        const int target = idx;
        bc.addToBot({[target](BattleContext &ctx) {
            auto &monster = ctx.monsters.arr[target];
            if (!monster.hasStatus<MS::FLIGHT>()) return;
            monster.decrementStatus<MS::FLIGHT>();
            if (!monster.hasStatus<MS::FLIGHT>() && monster.id == MonsterId::BYRD) {
                ctx.addToBot({[target](BattleContext &next) {
                    next.monsters.arr[target].setMove(MMID::BYRD_STUNNED);
                }});
            }
        }});

    } else if (damage < curHp && (hasStatus<MS::MALLEABLE>() || hasStatus<MS::REACTIVE>())) {
        if (hasStatus<MS::MALLEABLE>()) {
            const auto malleable = getStatus<MS::MALLEABLE>();
            bc.addToBot( Actions::MonsterGainBlock(this->idx, malleable) );
            setStatus<MS::MALLEABLE>(malleable+1);
        }
        if (hasStatus<MS::REACTIVE>()) {
            setStatus<MS::REACTIVE>(getStatus<MS::REACTIVE>() + 1);
            bc.addToBot(Actions::ReactiveRollMove(idx));
        }

    } else if (hasStatus<MS::ASLEEP>()) {
        // lagavulin
        setHasStatus<MS::ASLEEP>(false);
        decrementStatus<MS::METALLICIZE>(8);

    } else if (hasStatus<MS::SHIFTING>()) {
        addDebuff<MS::STRENGTH>(-damage);
        buff<MS::SHACKLED>(damage);
    }

    curHp -= damage;
    if (curHp <= 0) {
        curHp = 0;
        die(bc);
        block = 0; // AbstractMonster.damage clears block after the death callbacks.
    } else {
        onHpLost(bc, damage);
    }
}

void Monster::attacked(BattleContext &bc, int damage) {
    if (isDeadOrEscaped()) { // these probably shouldn't happen
        std::cerr << bc.seed << " was dead when attacked" << idx << "\n" << bc;
        assert(false);
        return;
    }

    if (damage < 0) {// these probably shouldn't happen
        damage = 0;
    }

    if (hasStatus<MS::INTANGIBLE>()) {
        if (damage > 0) {
            damage = 1;
        }
    }

    const bool hadBlock = block > 0;
    const int tempDamage = damage;
    damage = std::max(0, damage - block);
    block = std::max(0, block - tempDamage);

    if (hadBlock && block == 0 && bc.player.hasRelic<RelicId::HAND_DRILL>()) {
        bc.addToBot(Actions::DebuffEnemy<MS::VULNERABLE>(idx, 2, false) );
    }

    attackedUnblockedHelper(bc, damage);
}

void Monster::damageUnblockedHelper(BattleContext &bc, int damage) {

    if (hasStatus<MS::INVINCIBLE>()) {
        damage = std::min(damage, getStatus<MS::INVINCIBLE>());
        setStatus<MS::INVINCIBLE>(getStatus<MS::INVINCIBLE>() - damage);
    }
        if (hasStatus<MS::ASLEEP>()) {
        // lagavulin
        setHasStatus<MS::ASLEEP>(false);
        decrementStatus<MS::METALLICIZE>(8);
    }
    if (hasStatus<MS::SHIFTING>()) {
        addDebuff<MS::STRENGTH>(-damage);
        buff<MS::SHACKLED>(damage);
    }

    curHp = std::max(0, curHp-damage);
    if (curHp == 0) {
        die(bc);
        block = 0;
    } else {
        onHpLost(bc, damage);
    }
}

void Monster::damage(BattleContext &bc, int damage) {
    if (isDeadOrEscaped()) { // these probably shouldn't happen
        std::cerr << bc.seed << " was dead when damaged" << idx << "\n" << bc;
        assert(false);
        return;
    }

    if (damage < 0) {
        damage = 0;
    }

    if (hasStatus<MS::INTANGIBLE>()) { // this is probably wrong with potions
        if (damage > 0) {
            damage = 1;
        }
    }

    const bool hadBlock = block > 0;
    const int tempDamage = damage;
    damage -= block;
    block = std::max(0, block-tempDamage);

    if (hadBlock && block == 0 && bc.player.hasRelic<RelicId::HAND_DRILL>()) {
        bc.addToBot(Actions::DebuffEnemy<MS::VULNERABLE>(idx, 2, false) );
    }

    if (damage > 0) {
        damageUnblockedHelper(bc, damage);
    } else {
        return;
    }
}

void Monster::onHpLost(BattleContext &bc, int amount) {
    const bool atOrBelowHalf = curHp <= maxHp/2;
    switch (id) {
        case MonsterId::ACID_SLIME_L:
            if (atOrBelowHalf) {
                moveHistory[0] = MMID::ACID_SLIME_L_SPLIT;
            }
            break;

        case MonsterId::SLIME_BOSS:
            if (atOrBelowHalf) {
                moveHistory[0] = MMID::SLIME_BOSS_SPLIT;
            }
            break;

        case MonsterId::SPIKE_SLIME_L:
            if (atOrBelowHalf) {
                moveHistory[0] = MMID::SPIKE_SLIME_L_SPLIT;
            }
            break;

        case MonsterId::THE_GUARDIAN:
            if (hasStatus<MS::MODE_SHIFT>()) {
                const int newModeShiftAmount = getStatus<MS::MODE_SHIFT>() - amount;
                if (newModeShiftAmount <= 0) {
                    removeStatus<MS::MODE_SHIFT>();
                    setMove(MMID::THE_GUARDIAN_DEFENSIVE_MODE);
                    bc.addToBot( Actions::MonsterGainBlock(idx, 20) );
                } else {
                    setStatus<MS::MODE_SHIFT>(newModeShiftAmount);
                }
            }
            break;

        default:
            break;
    }
}

void Monster::removeDebuffs() {
    if (getStatus<MS::STRENGTH>() < 0) {
        setStatus<MS::STRENGTH>(0);
    }

    removeStatus<MS::BLOCK_RETURN>();
    removeStatus<MS::CHOKED>();
    removeStatus<MS::CORPSE_EXPLOSION>();
    removeStatus<MS::LOCK_ON>();
    removeStatus<MS::MARK>();
    removeStatus<MS::POISON>();
    removeStatus<MS::SHACKLED>();
    removeStatus<MS::VULNERABLE>();
    removeStatus<MS::WEAK>();
}

void Monster::resetAllStatusEffects() {
    statusBits = 0;
    setStatus<MS::STRENGTH>(0);
    block = 0;
}

// this is calculated at the player when the damage occurs in game, consider testing whether there are any scenarios when it can't be done before
int Monster::getIntentDamage(const BattleContext &bc, int baseDamage) const {
    if ((isDeadOrEscaped() || isDeadB) && inactiveIntentDamage >= 0) {
        return inactiveIntentDamage;
    }
    return baseDamage > 0 ? calculateDamageToPlayer(bc, baseDamage) : 0;
}

int Monster::calculateDamageToPlayer(const BattleContext &bc, int baseDamage) const {
    const auto &p = bc.player;
    auto damage = static_cast<float>(baseDamage + getStatus<MS::STRENGTH>());

    if (p.hasStatus<PS::SURROUNDED>()) { // todo this is probably wrong
        const bool facingSelf = p.lastTargetedMonster == idx || bc.monsters.arr[p.lastTargetedMonster].isDeadOrEscaped();
        if (!facingSelf) {
            damage *= 1.5;
        }
    }

    if (hasStatus<MS::WEAK>()) {
        if (p.hasRelic<RelicId::PAPER_KRANE>()) {
            damage *= 0.6f;
        } else {
            damage *= 0.75f;
        }
    }

    if (p.hasStatus<PS::VULNERABLE>()) {
        if (p.hasRelic<RelicId::ODD_MUSHROOM>()) {
            damage *= 1.25f;
        } else {
            damage *= 1.5f;
        }
    }

    if (p.stance == Stance::WRATH) {
        damage *= 2;
    }

    // apply backIdx attack for spire guard elites

    if (p.hasStatus<PS::INTANGIBLE>()) {
        damage = std::min(damage, static_cast<float>(1));
    }

    return std::max(static_cast<int>(std::floor(damage)), 0);
}

void Monster::attackPlayerHelper(BattleContext &bc, int baseDamage, int times) {
    const int damage = calculateDamageToPlayer(bc, baseDamage);

    for (int i = 0; i < times; i++) {
        bc.addToBot(Actions::AttackPlayer(idx, damage));
    }
}

bool Monster::firstTurn() const {
    return moveHistory[0] == MMID::INVALID;
}

bool Monster::lastMove(MMID moveId) const {
    return moveHistory[0] == moveId;
}

bool Monster::lastMoveBefore(MMID moveId) const {
    return moveHistory[1] == moveId;
}

bool Monster::lastTwoMoves(MMID moveId) const {
    return moveHistory[0] == moveId && moveHistory[1] == moveId;
}

bool Monster::eitherLastTwo(MonsterMoveId moveId) const {
    return moveHistory[0] == moveId || moveHistory[1] == moveId;
}

void Monster::rollMove(BattleContext &bc) {
    auto miscInfoCopy = miscInfo;

    const auto move = getMoveForRoll(bc, miscInfoCopy, bc.aiRng.random(99));

    miscInfo = miscInfoCopy;
    setMove(move);
}

void Monster::setMove(MMID moveId) {
    inactiveIntentDamage = -1;
    moveHistory[2] = moveHistory[1];
    moveHistory[1] = moveHistory[0];
    moveHistory[0] = moveId;
}

namespace sts {

    std::ostream& printIfHaveStatus(std::ostream &os, const Monster &m, MonsterStatus s, bool &havePrint) {
        if (!m.getStatusInternal(s)) {
            return os;
        }

        if (havePrint) {
            os << ", ";
        }
        havePrint = true;

        os << '(' << enemyStatusStrings[(int)s];
        if (!isBooleanPower(s)) {
            os << "," << m.getStatusInternal(s);
        }
        os << ")";
        return os;
    }



    std::ostream &operator<<(std::ostream &os, const Monster &m) {
        os << "{";
        os << m.idx << " " << sts::monsterIdStrings[(int) m.id]
           << " hp:(" << m.curHp << "/" << m.maxHp << ")"
           << " block:(" << m.block << ") statusEffects:{";

        bool havePrint = false;
        printIfHaveStatus(os, m, MS::STRENGTH, havePrint);
        for (int i = static_cast<int>(MS::ARTIFACT); i <= static_cast<int>(MS::STASIS); ++i) {
            auto s = static_cast<MS>(i);
            if (s != MonsterStatus::STRENGTH) {
                printIfHaveStatus(os, m, s, havePrint);
            }
        }
        os << "}";

        os << " halfDead: " << m.halfDead
           << ", moveHistory: { " << monsterMoveStrings[(int) m.moveHistory[0]] << ", " << monsterMoveStrings[(int) m.moveHistory[1]] << ", " << monsterMoveStrings[(int) m.moveHistory[2]] << "}"
           << " nextActionDamage: " << " miscInfo: " << m.miscInfo << " uniquePower0: " << m.uniquePower0 << " uniquePower1: " << m.uniquePower1
            << "}";
        return os;
    }

}
