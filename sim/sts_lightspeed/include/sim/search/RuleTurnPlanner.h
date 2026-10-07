#ifndef STS_RULE_TURN_PLANNER_H
#define STS_RULE_TURN_PLANNER_H

#include "sim/search/Action.h"
#include "sim/search/Nosl.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace sts::search::ruleturn {

// The Python planner remains the reference. Native execution preserves its
// double arithmetic order, serial tie breaks, per-root quotas and NOSL worlds.
inline std::string normalized(std::string s) {
    std::string out;
    for (unsigned char c : s) if (c != ' ' && c != '_') out += char(std::tolower(c));
    return out;
}

struct Tables {
    std::array<bool, std::size(cardEnumStrings)> junk{}, harmless{};
    std::array<bool, static_cast<int>(PlayerStatus::THE_BOMB)+1> harmlessPower{};
    std::unordered_map<std::string, PlayerStatus> powers;
    Tables() {
        const std::unordered_set<std::string> j = {"WOUND","DAZED","BURN","VOID","SLIMED","INJURY",
            "DOUBT","SHAME","PAIN","REGRET","DECAY","NORMALITY","WRITHE"};
        const std::unordered_set<std::string> h = {"DEFEND_RED","ARMAMENTS","BATTLE_TRANCE","BLOODLETTING",
            "BURNING_PACT","DISARM","DUAL_WIELD","ENTRENCH","FLEX","HAVOC","INTIMIDATE","POWER_THROUGH",
            "RAGE","SECOND_WIND","SEEING_RED","SENTINEL","SHOCKWAVE","SHRUG_IT_OFF","TRUE_GRIT",
            "WARCRY","IMPERVIOUS","GHOSTLY_ARMOR","LIMIT_BREAK","ASCENDERS_BANE","CLUMSY",
            "CURSE_OF_THE_BELL","NECRONOMICURSE","PARASITE","PRIDE","WRITHE"};
        const std::unordered_set<std::string> hp = {"strength","dexterity","artifact","weak","vulnerable",
            "frail","metallicize","platedarmor","barricade","berserk","demonform","feelnopain",
            "darkembrace","corruption","evolve","rupture","rage","nodraw","drawreduction","entangled",
            "confused","intangible","buffer","energized","noblock","hex","constricted","drawcardnextturn",
            "doubletap","doubledamage","vigor","losestrength","dexteritydown"};
        for (std::size_t i=0; i<junk.size(); ++i) {
            junk[i] = j.count(cardEnumStrings[i]);
            harmless[i] = junk[i] || h.count(cardEnumStrings[i]);
        }
        for (int i=1; i<=static_cast<int>(PlayerStatus::THE_BOMB); ++i) {
            auto name=normalized(playerStatusStrings[i]);
            powers[name]=static_cast<PlayerStatus>(i);
            harmlessPower[i]=hp.count(name);
        }
    }
};
inline const Tables &tables() { static const Tables value; return value; }

struct Config { int worlds=2, nodes=8192, beam=8, depth=16, turns=1; bool timeClock=false; };
struct Stats {
    int nativeTransitions=0, expandedNodes=0, fallbackLeaves=0, secondTurnEntries=0;
    int completedHorizonLeaves=0, maxDepth=0, winningSteps=0;
    bool budgetExhausted=false;
    void add(const Stats &s) {
        nativeTransitions+=s.nativeTransitions; expandedNodes+=s.expandedNodes;
        fallbackLeaves+=s.fallbackLeaves; secondTurnEntries+=s.secondTurnEntries;
        completedHorizonLeaves+=s.completedHorizonLeaves;
        maxDepth=std::max(maxDepth,s.maxDepth); budgetExhausted|=s.budgetExhausted;
    }
};
struct RootResult { std::uint32_t bits; int deaths=0,wins=0,unresolved=0,winningSteps=0; double mean=0; };
struct Result { std::uint32_t action=0; Stats stats; std::vector<RootResult> roots; int budget=0; double seconds=0; };
using Score=std::pair<int,double>;

struct Planner {
    Config config;
    std::function<std::vector<Action>(const BattleContext &)> legal;
    std::function<std::uint32_t(const BattleContext &,const BattleContext &,std::uint32_t)> remap;
    int (*observePower)(const Player &,PlayerStatus);
    int power(const Player &p,const char *name) const {
        auto it=tables().powers.find(name);
        return it==tables().powers.end() ? 0 : observePower(p,it->second);
    }
    bool noDamage(const BattleContext &b,bool outputRelic) const {
        if (outputRelic || b.outcome!=Outcome::UNDECIDED || b.inputState==InputState::CARD_SELECT) return false;
        auto harmless=[](const auto &c) {return tables().harmless[static_cast<int>(c.getId())];};
        for (int i=0;i<b.cards.cardsInHand;++i) if (!harmless(b.cards.hand[i])) return false;
        for (const auto &c:b.cards.drawPile) if (!harmless(c)) return false;
        for (const auto &c:b.cards.discardPile) if (!harmless(c)) return false;
        for (int i=1;i<=static_cast<int>(PlayerStatus::THE_BOMB);++i)
            if (!tables().harmlessPower[i] && observePower(b.player,static_cast<PlayerStatus>(i))>0) return false;
        if (b.player.bomb1>0 || b.player.bomb2>0 || b.player.bomb3>0) return false;
        for (int i=0;i<b.potionCapacity;++i)
            if (b.potions[i]!=Potion::INVALID && b.potions[i]!=Potion::EMPTY_POTION_SLOT) return false;
        for (int i=0;i<b.monsters.monsterCount;++i) {
            const auto &m=b.monsters.arr[i];
            if (m.curHp<=0 || m.isDeadOrEscaped()) continue;
            if (m.getStatusInternal(MonsterStatus::POISON)>0 || m.getStatusInternal(MonsterStatus::CHOKED)>0 ||
                m.getStatusInternal(MonsterStatus::FADING)>0) return false;
            if (m.id==MonsterId::TRANSIENT || m.id==MonsterId::DAGGER || m.id==MonsterId::EXPLODER) return false;
        }
        return true;
    }
    double horizon(const BattleContext &b) const {
        int remaining=0;
        for (int i=0;i<b.monsters.monsterCount;++i) {
            const auto &m=b.monsters.arr[i];
            if (m.curHp>0 && !m.isDeadOrEscaped()) remaining+=m.curHp;
        }
        return std::min(5.,std::max(1.,remaining/std::max(25.,double(25+3*power(b.player,"strength")))));
    }
    double value(const BattleContext &b,bool partial,bool outputRelic,double h) const {
        const auto &p=b.player;
        if (p.curHp<=0 || b.outcome==Outcome::PLAYER_LOSS) return -100000.;
        int remaining=0,bodies=0,potions=0,junk=0,incoming=0;
        for (int i=0;i<b.monsters.monsterCount;++i) {
            const auto &m=b.monsters.arr[i];
            if (m.id==MonsterId::AWAKENED_ONE && m.halfDead) {remaining+=m.maxHp; ++bodies; continue;}
            if (m.curHp<=0 || m.isDeadOrEscaped()) continue;
            if (m.id==MonsterId::AWAKENED_ONE && m.getStatusInternal(MonsterStatus::CURIOSITY)>0) {
                remaining+=m.curHp+m.maxHp; ++bodies;
            } else if (m.id==MonsterId::ACID_SLIME_L || m.id==MonsterId::SPIKE_SLIME_L) {
                remaining+=m.curHp+std::min(m.curHp,m.maxHp/2); bodies+=2;
            } else if (m.id==MonsterId::SLIME_BOSS) {
                const int split=std::min(m.curHp,m.maxHp/2);
                remaining+=m.curHp+split+2*(split/2); bodies+=4;
            } else {remaining+=m.curHp; ++bodies;}
        }
        auto q=[&](const char *name) {return power(p,name);};
        double v=1.3*p.curHp+12*std::log(std::max(.01,double(p.curHp)/p.maxHp))+1.8*p.maxHp+.025*p.gold;
        for (int i=0;i<b.potionCapacity;++i) potions+=b.potions[i]!=Potion::INVALID && b.potions[i]!=Potion::EMPTY_POTION_SLOT;
        v+=5*potions;
        if (b.outcome==Outcome::PLAYER_VICTORY) return v+1000;
        if (noDamage(b,outputRelic)) v-=10000;
        v-=.26*remaining+10*bodies;
        for (int i=0;i<b.monsters.monsterCount;++i) {
            const auto &m=b.monsters.arr[i];
            if (m.curHp<=0 || m.isDeadOrEscaped()) continue;
            v-=1.6*h*m.getStatusInternal(MonsterStatus::STRENGTH);
            v-=h*m.getStatusInternal(MonsterStatus::RITUAL);
            v+=3*std::min(h,double(m.getStatusInternal(MonsterStatus::WEAK)));
            v+=3*std::min(h,double(m.getStatusInternal(MonsterStatus::VULNERABLE)));
            v-=.6*std::min(m.block,30);
            if (partial) {
                const auto dmg=m.getMoveBaseDamage(b);
                incoming+=std::max(0,dmg.damage>0 ? m.calculateDamageToPlayer(b,dmg.damage):0)*std::max(0,dmg.attackCount);
            }
        }
        v+=h*(2.3*q("strength")+1.8*q("dexterity")+.8*q("metallicize")+.8*q("platedarmor")+1.2*q("combust")+2*q("berserk"));
        v+=1.2*h*h*q("demonform");
        v+=h*(2*q("feelnopain")+5*q("darkembrace")+6*q("corruption"));
        v-=2*std::min(h,double(q("weak"))); v-=2*std::min(h,double(q("vulnerable")));
        if (q("barricade")) v+=.55*std::min(150,p.block)+4*h;
        for (int i=0;i<b.cards.cardsInHand;++i) junk+=tables().junk[static_cast<int>(b.cards.hand[i].getId())];
        for (const auto &c:b.cards.drawPile) junk+=tables().junk[static_cast<int>(c.getId())];
        for (const auto &c:b.cards.discardPile) junk+=tables().junk[static_cast<int>(c.getId())];
        v-=.7*junk;
        if (config.timeClock && b.postEnemyTurnBoundary) {
            for (int i=0;i<b.monsters.monsterCount;++i) {
                const auto &m=b.monsters.arr[i];
                if (m.id==MonsterId::TIME_EATER && m.curHp>0 && !m.isDeadOrEscaped()) {
                    int n=m.getStatusInternal(MonsterStatus::TIME_WARP);
                    if (n>7) {v-=3.*(n-7)*(n-7); break;}
                }
            }
        }
        if (partial) {v-=1.3*std::max(0,incoming-p.block); v+=2*p.energy+.4*b.cards.cardsInHand;}
        if (!std::isfinite(v)) throw std::runtime_error("rule planner value must be finite");
        return v;
    }
    std::pair<Score,Stats> root(const BattleContext &branch,Action action,int quota,bool outputRelic,double h,int finalTurn) const {
        struct Node {double priority; int serial,depth; std::unique_ptr<BattleContext> state;};
        auto less=[](const Node &a,const Node &b) {return std::tie(a.priority,a.serial)<std::tie(b.priority,b.serial);};
        auto greater=[&](const Node &a,const Node &b) {return less(b,a);};
        std::vector<Node> frontier;
        Stats stats; int serial=0; bool hasBest=false,hasPartial=false; Score best; double bestPartial=0;
        auto take=[&](const BattleContext &state,Action a,int depth) {
            ++stats.nativeTransitions; stats.maxDepth=std::max(stats.maxDepth,depth);
            auto child=std::make_unique<BattleContext>(state);
#ifdef STS_ENABLE_RULE_TRACE
            child->atomicStepObserver={};
#endif
            if (config.turns==1 || state.turn>=finalTurn) child->stopAfterEnemyTurn=true;
            a.execute(*child);
            if (config.turns==2 && state.turn<finalTurn && child->outcome==Outcome::UNDECIDED && child->turn>=finalTurn)
                ++stats.secondTurnEntries;
            if (child->outcome!=Outcome::UNDECIDED || child->postEnemyTurnBoundary) {
                stats.completedHorizonLeaves+=child->postEnemyTurnBoundary;
                const int rank=child->player.curHp<=0 || child->outcome==Outcome::PLAYER_LOSS ? -1 : child->outcome==Outcome::PLAYER_VICTORY ? 1 : 0;
                Score score{rank,value(*child,false,outputRelic,h)};
                if (!hasBest || score>best) {best=score; hasBest=true; stats.winningSteps=rank==1?depth:0;}
                else if (score==best && rank==1) stats.winningSteps=std::min(stats.winningSteps,depth);
                return;
            }
            double partial=value(*child,true,outputRelic,h);
            if (!hasPartial || partial>bestPartial) bestPartial=partial;
            hasPartial=true;
            frontier.push_back({-partial,++serial,depth,std::move(child)});
            std::push_heap(frontier.begin(),frontier.end(),greater);
        };
        take(branch,action,1);
        while (!frontier.empty() && stats.nativeTransitions<quota) {
            std::pop_heap(frontier.begin(),frontier.end(),greater);
            Node node=std::move(frontier.back()); frontier.pop_back(); ++stats.expandedNodes;
            auto actions=legal(*node.state);
            if (config.turns==2 && node.depth>=config.depth) continue;
            const auto end=std::find_if(actions.begin(),actions.end(),[](const auto &a){return a.bits==(4u<<29);});
            if (end!=actions.end()) take(*node.state,*end,node.depth+1);
            if (node.depth>=config.depth) continue;
            for (const auto &a:actions) {
                if (a.bits==(4u<<29)) continue;
                if (stats.nativeTransitions>=quota) {stats.budgetExhausted=true; break;}
                take(*node.state,a,node.depth+1);
            }
            if (frontier.size()>static_cast<std::size_t>(config.beam)) {
                std::sort(frontier.begin(),frontier.end(),less); frontier.resize(config.beam);
                std::make_heap(frontier.begin(),frontier.end(),greater);
            }
        }
        stats.budgetExhausted|=!frontier.empty();
        if (!hasBest) {
            ++stats.fallbackLeaves;
            if (!hasPartial) throw std::runtime_error("rule planner produced no leaf");
            best={0,bestPartial};
        }
        return {best,stats};
    }
    Result choose(const BattleContext &battle,const std::vector<std::uint64_t> &nonces,bool outputRelic) const {
        const auto started=std::chrono::steady_clock::now();
        auto actions=legal(battle);
        if (actions.empty() || battle.postEnemyTurnBoundary) throw std::invalid_argument("rule search requires a live combat decision");
        Result out; out.budget=std::max(config.nodes,(config.turns+1)*static_cast<int>(actions.size()));
        for (auto a:actions) out.roots.push_back({a.bits});
        const int quota=out.budget/static_cast<int>(actions.size()),finalTurn=battle.turn+config.turns-1;
        const auto task=battle.cardSelectInfo.cardSelectTask;
        const bool selectDraw=battle.inputState==InputState::CARD_SELECT &&
            (task==CardSelectTask::SECRET_TECHNIQUE || task==CardSelectTask::SECRET_WEAPON || task==CardSelectTask::SEEK);
        const double h=horizon(battle);
        for (auto nonce:nonces) {
            BattleContext sample(battle);
#ifdef STS_ENABLE_RULE_TRACE
            sample.atomicStepObserver={};
#endif
            noslRedeterminize(sample,nonce);
            for (std::size_t i=0;i<actions.size();++i) {
                const auto action=selectDraw ? Action(remap(sample,battle,actions[i].bits)):actions[i];
                auto result=root(sample,action,quota,outputRelic,h,finalTurn); auto &r=out.roots[i];
                r.deaths+=result.first.first==-1; r.wins+=result.first.first==1;
                r.mean+=result.first.second; r.unresolved+=result.second.fallbackLeaves;
                r.winningSteps+=result.second.winningSteps; out.stats.add(result.second);
            }
        }
        for (auto &r:out.roots) r.mean/=nonces.size();
        auto priority=[](const RootResult &r){return std::make_tuple(-r.deaths,r.wins,r.mean,-r.winningSteps,-std::int64_t(r.bits));};
        const auto best=std::max_element(out.roots.begin(),out.roots.end(),[&](const auto &a,const auto &b){return priority(a)<priority(b);});
        out.action=best->bits;
        out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        return out;
    }
};
} // namespace sts::search::ruleturn
#endif
