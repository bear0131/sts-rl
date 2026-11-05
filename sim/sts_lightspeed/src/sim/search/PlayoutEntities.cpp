#include "sim/search/PlayoutEntities.h"
#include "sim/search/CombatChoices.h"
#include "game/GameContext.h"
#include <algorithm>
#include <numeric>
#include <tuple>

namespace sts::search {
namespace {
using K = PlayoutKind;
using Z = PlayoutZone;
PlayoutEntity row(K kind, int id, Z zone, int monster=-1) {
    if (id < 0 || id >= PLAYOUT_ENTITY_STRIDE) throw std::out_of_range("public playout entity ID");
    return {int(kind)*PLAYOUT_ENTITY_STRIDE + id, int(zone), monster, {}};
}
template<class T> float f(T x) { return static_cast<float>(x); }
}

std::vector<PlayoutEntity> playoutInventory(const GameContext &gc) {
    std::vector<PlayoutEntity> rows;
    int order=0;
    for (const auto &r: gc.relics.relics) {
        auto v=row(K::Relic,int(r.id),Z::OwnedRelic);
        v.numbers[0]=f(++order)/64; v.numbers[1]=f(r.data)/10;
        rows.push_back(v);
    }
    // The publicly inspectable master deck is a multiset, not RNG order.
    // Pair every bottle with its actual card without exposing physical UID.
    for (int i=0;i<gc.deck.size();++i) {
        const auto &card=gc.deck.cards[i];
        auto v=row(K::Card,int(card.id),Z::MasterDeck);
        v.numbers[0]=f(card.getUpgraded())/10; v.numbers[1]=f(card.misc)/100;
        for (int j=0;j<3;++j) v.numbers[2+j]=gc.deck.bottleIdxs[j]==i;
        rows.push_back(v);
    }
    std::sort(rows.begin()+order,rows.end(),[](const auto &a,const auto &b){
        return std::tie(a.id,a.numbers)<std::tie(b.id,b.numbers);
    });
    return rows;
}

void playoutEntities(const BattleContext &bc, int masterDeckSize,
                     const std::vector<SmallActionFeatures> &actions, PlayoutEntityObservation &out) {
    out.entities.clear(); out.sources.clear(); out.targets.clear();
    auto add=[&](PlayoutEntity v){int i=int(out.entities.size());out.entities.push_back(v);return i;};
    const auto &p=bc.player; const auto &c=bc.cards; const auto &s=bc.cardSelectInfo;
    const bool selection=bc.inputState==InputState::CARD_SELECT;
    if(selection && (s.selectedCount<0 || s.selectedCount>CardSelectInfo::GRID_PICK_CAPACITY))
        throw std::logic_error("invalid pending selected card count");
    auto battle=row(K::Battle,int(bc.encounter),Z::Battle);
    battle.numbers={f(bc.ascension)/20,f(bc.floorNum)/60,f(bc.turn)/50,f(bc.inputState)/8,
        f(bc.potionCapacity)/5,f(bc.potionCount)/5,f(bc.monsters.monsterCount)/5,
        f(bc.monsters.monstersAlive)/5,f(bc.postEnemyTurnBoundary),f(bc.pocketwatchCardsAtBoundary)/10,
        f(bc.endTurnQueued),f(bc.turnHasEnded),f(bc.skipMonsterTurn),f(c.cardsInHand)/10,
        f(c.knownDrawTop)/40,f(bc.outcome)/4}; add(battle);
    auto pl=row(K::Player,int(p.cc),Z::Player);
    pl.numbers={f(p.curHp)/200,f(p.maxHp)/200,f(p.block)/100,f(p.energy)/10,f(p.gold)/500,
        f(p.energyPerTurn)/10,f(p.cardDrawPerTurn)/10,f(p.stance)/5,f(p.orbSlots)/10,
        f(p.lastTargetedMonster)/5,f(p.strength)/20,f(p.dexterity)/20,f(p.focus)/20,
        f(p.artifact)/5,f(p.bloodied),f(p.redSkullActive)}; add(pl);
    const int counters[]={p.happyFlowerCounter,p.incenseBurnerCounter,p.inkBottleCounter,p.inserterCounter,
        p.nunchakuCounter,p.penNibCounter,p.sundialCounter,p.neowLamentRemaining,p.haveUsedNecronomiconThisTurn,
        p.combustHpLoss,p.devaFormEnergyPerTurn,p.echoFormCardsDoubled,p.panacheCounter,p.cardsPlayedThisTurn,
        p.attacksPlayedThisTurn,p.skillsPlayedThisTurn,int(p.orangePelletsCardTypesPlayed.to_ulong()),
        p.cardsDiscardedThisTurn,p.lastAttackUnblockedDamage,p.timesDamagedThisCombat,p.feelNoPainBeforeDarkEmbrace};
    for (int i=0;i<int(sizeof(counters)/sizeof(counters[0]));++i) if (counters[i]) {
        auto v=row(K::Counter,i,Z::Player);v.numbers[0]=f(counters[i])/10;add(v);
    }
    for(int i=1;i<=int(PS::THE_BOMB);++i) if(p.hasStatusRuntime(PS(i))) {
        auto v=row(K::Power,i,Z::Player);v.numbers[0]=f(p.getStatusRuntime(PS(i)))/10;
        v.numbers[1]=i<32 && ((p.justAppliedBits>>i)&1u);add(v);
    }
    for (int i=0;i<int(p.bombs.size());++i) {
        auto v=row(K::Counter,32,Z::Player);v.numbers[0]=f(p.bombs[i].turns)/10;
        v.numbers[1]=f(p.bombs[i].damage)/100;v.numbers[2]=f(i+1)/10;add(v);
    }
    for(int i=0;i<128;++i) if(((i<64?p.relicBits0:p.relicBits1)>>(i%64))&1ull)
        add(row(K::Relic,i,Z::ActiveRelic));
    auto order=[&](const auto &items,int group){for(int i=0;i<int(items.size());++i){
        auto v=row(K::Order,int(items[i]),Z::Player);v.numbers[0]=f(group);
        v.numbers[1]=f(i+1)/32;add(v);
    }};
    order(bc.onUseRelicOrder,1);order(bc.onEndTurnRelicOrder,2);order(bc.onPostDrawRelicOrder,3);
    order(p.endTurnPowerOrder,4);
    std::array<int,5> pot{};
    for(int i=0;i<bc.potionCapacity;++i){auto v=row(K::Potion,int(bc.potions[i]),Z::Potion);
        v.numbers[0]=f(i+1)/5;pot[i]=add(v);}
    const bool dome=p.hasRelic<R::RUNIC_DOME>();
    for(int i=0;i<5;++i){
        const auto &m=bc.monsters.arr[i];auto v=row(K::Monster,int(m.id),Z::Monster,i);
        if(m.id!=MonsterId::INVALID){
            v.numbers={f(m.curHp)/200,f(m.maxHp)/200,f(m.block)/100,f(m.strength)/20,
                f(m.isEscapingB),f(m.isDyingB),f(m.isDeadB),f(m.halfDead),f(m.isTargetable()),
                f(m.isDeadOrEscaped()),f(m.isAlive()),f(m.isEscaping()),0,0,0,0};
            // A concealed intent may not leak through escapeNext, displayed
            // inactive damage or its cached RNG-selected base damage.
            if(!dome){v.numbers[12]=f(m.escapeNext);v.numbers[13]=f(m.inactiveIntentDamage)/100;}
        }
        add(v);if(m.id==MonsterId::INVALID)continue;
        for(int j=0;j<int(MS::INVALID);++j) if(m.hasStatusInternal(MS(j)) || (MS(j)==MS::STRENGTH && m.strength)){
            auto q=row(K::MonsterPower,j,Z::Monster,i);q.numbers[0]=f(m.getStatusInternal(MS(j)))/10;
            if(MS(j)==MS::VULNERABLE)q.numbers[1]=m.wasJustApplied<MS::VULNERABLE>();
            if(MS(j)==MS::WEAK)q.numbers[1]=m.wasJustApplied<MS::WEAK>();
            if(MS(j)==MS::RITUAL)q.numbers[1]=m.wasJustApplied<MS::RITUAL>();add(q);
        }
        for(int h=dome?1:0;h<3;++h){auto q=row(K::Move,int(m.moveHistory[h]),Z::Monster,i);
            q.numbers[0]=f(h);if(!h){const auto d=m.getMoveBaseDamage(bc);
                q.numbers[1]=f(d.damage)/100;q.numbers[2]=f(m.getIntentDamage(bc,d.damage))/100;
                q.numbers[3]=f(d.attackCount)/10;}add(q);}
        if(dome)continue; // Engine move-state latches can be set while rolling a concealed intent.
        auto q=row(K::Counter,33,Z::Monster,i);
        switch(m.id){
        case MonsterId::LOOTER:case MonsterId::MUGGER:
            q.numbers[1]=f(m.thiefSlashCount)/10;q.numbers[2]=f(m.thiefGoldAmount)/100;
            [[fallthrough]];
        case MonsterId::LAGAVULIN:case MonsterId::GREMLIN_WIZARD:case MonsterId::WRITHING_MASS:
        case MonsterId::TIME_EATER:case MonsterId::RED_SLAVER:case MonsterId::BRONZE_ORB:
        case MonsterId::SPIKER:case MonsterId::THE_CHAMP:case MonsterId::BRONZE_AUTOMATON:
        case MonsterId::BOOK_OF_STABBING:case MonsterId::THE_GUARDIAN:case MonsterId::GIANT_HEAD:
        case MonsterId::AWAKENED_ONE:case MonsterId::HEXAGHOST:
            q.numbers[0]=f(m.miscInfo)/32;add(q);break;
        default:break; // Never encode concealed Louse bite/Darkling nip rolls or an AI target slot.
        }
    }
    auto card=[&](const CardInstance &x,Z zone,int position,int picked=0){
        auto v=row(K::Card,int(x.id),zone);
        v.numbers={f(x.getUpgradeCount())/10,f(x.specialData)/100,f(x.cachedBaseDamage)/100,
            f(x.cost)/5,f(x.costForTurn)/5,f(x.freeToPlayOnce),f(x.retain),f(x.exhaust),
            f(position)/40,f(picked)/2,f(x.uniqueId>=0 && x.uniqueId<masterDeckSize),0,0,0,0,0};
        return v;
    };
    std::array<std::vector<int>,6> mapping;
    for(int i=0;i<c.cardsInHand;++i)mapping[1].push_back(add(card(c.hand[i],Z::Hand,i+1)));
    const int draw=int(c.drawPile.size());
    const int known=p.hasRelic<R::FROZEN_EYE>()?draw:std::clamp(c.knownDrawTop,0,draw);
    std::vector<std::pair<PlayoutEntity,int>> hidden;hidden.reserve(draw-known);
    mapping[2].resize(draw);
    for(int i=0;i<draw-known;++i)hidden.emplace_back(card(c.drawPile[i],Z::HiddenDraw,0),i);
    std::sort(hidden.begin(),hidden.end(),[](const auto &a,const auto &b){
        return std::tie(a.first.id,a.first.numbers)<std::tie(b.first.id,b.first.numbers);
    });
    for(const auto &pair:hidden)mapping[2][pair.second]=add(pair.first);
    for(int i=draw-1;i>=draw-known;--i)mapping[2][i]=add(card(c.drawPile[i],Z::KnownDraw,draw-i));
    for(int i=0;i<int(c.discardPile.size());++i){int picked=0;
        if(selection)for(int j=0;j<s.selectedCount;++j)if(c.discardPile[i].uniqueId==s.selectedUids[j])picked=j+1;
        mapping[3].push_back(add(card(c.discardPile[i],Z::Discard,i+1,picked)));}
    for(int i=0;i<int(c.exhaustPile.size());++i)mapping[4].push_back(add(card(c.exhaustPile[i],Z::Exhaust,i+1)));
    for(int i=0;i<2;++i)if(c.stasisCards[i].id!=CardId::INVALID)add(card(c.stasisCards[i],Z::Stasis,i+1));
    int zone=0;
    if(selection){
        const auto choices=combatChoices(bc,CombatChoiceContract::NativeHandles);zone=choices.zone;
        auto v=row(K::Selection,int(s.cardSelectTask),Z::Pending);
        v.numbers={f(s.canPickZero),f(s.canPickAnyNumber),f(s.pickCount)/10,f(s.selectedCount)/10,f(zone)/7,f(s.data0)/10};
        const auto task=s.cardSelectTask;
        if(task==CardSelectTask::DISCOVERY)v.numbers[6]=s.discoveryFreeForTurn;
        add(v);
        if(task==CardSelectTask::DISCOVERY || task==CardSelectTask::CODEX)
            for(int i=0;i<3;++i)mapping[5].push_back(add(card(bc.makeCardCopy(s.cards[i]),Z::Generated,i+1)));
        if(s.origin==SelectOrigin::Card && int(bc.curCardQueueItem.card.id)==s.originId)
            add(card(bc.curCardQueueItem.card,Z::Pending,0));
        else if(s.origin==SelectOrigin::Potion)add(row(K::Potion,s.originId,Z::Pending));
        else if(s.origin==SelectOrigin::Relic)add(row(K::Relic,s.originId,Z::Pending));
        else throw std::logic_error("public playout selection lacks a matching recorded origin");
    }
    for(const auto &features:actions){
        Action a(features.bits);std::array<int,SMALL_KEYS> ids;ids.fill(-1);int target=-1;
        const auto kind=a.getActionType();
        if(kind==ActionType::CARD){int i=a.getSourceIdx();ids[0]=mapping[1].at(i);
            if(c.hand[i].requiresTarget())target=a.getTargetIdx();}
        else if(kind==ActionType::POTION){int i=a.getSourceIdx();ids[0]=pot.at(i);
            if(potionRequiresTarget(bc.potions[i]))target=a.getTargetIdx();}
        else if(kind==ActionType::SINGLE_CARD_SELECT){int i=a.getSelectIdx();
            if(!(s.cardSelectTask==CardSelectTask::CODEX && i==3))ids[0]=mapping.at(zone).at(i);}
        else if(kind==ActionType::MULTI_CARD_SELECT){int j=0;
            for(int i:a.getSelectedIdxs())ids.at(j++)=mapping[1].at(i);}
        if(target < -1 || target >=5)throw std::logic_error("invalid public playout target slot");
        out.sources.push_back(ids);out.targets.push_back(target);
    }
}
}
