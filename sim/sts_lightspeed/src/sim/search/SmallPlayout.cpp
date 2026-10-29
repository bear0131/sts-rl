#include "sim/search/SmallPlayout.h"
#include "sim/search/CombatChoices.h"
#include "sim/search/Nosl.h"
#include "sim/search/NoslCombatTrainer.h"
#include "sim/search/BattleScumSearcher2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

namespace sts::search {
namespace {
constexpr int CARD_KEYS = 2 * static_cast<int>(CardId::ZAP);
constexpr int POTION_FIRST = static_cast<int>(Potion::AMBROSIA);
constexpr int POTION_KEYS = static_cast<int>(Potion::WEAK_POTION) - POTION_FIRST + 1;
int cardKey(const CardInstance &c) {
    if (c.id == CardId::INVALID) throw std::logic_error("invalid compact card");
    return 3 + 2 * (static_cast<int>(c.id) - 1) + int(c.getUpgradeCount() > 0);
}
float incoming(const BattleContext &bc, const Monster &m) {
    if (!m.isTargetable()) return 0;
    const auto damage = m.getMoveBaseDamage(bc);
    return float(m.getIntentDamage(bc, damage.damage) * damage.attackCount);
}
void addCard(SmallActionFeatures &f, const CardInstance &c) {
    f.keys.push_back(cardKey(c));
    f.x[5] += float(c.costForTurn) / 5;
    f.x[6] += float(c.getUpgradeCount()) / 10;
    f.x[7] += float(c.getType()) / 5;
    f.x[8] += float(c.specialData) / 100;
}
}
int smallKeyCount() { return 3 + CARD_KEYS + POTION_KEYS; }

SmallFeatures smallFeatures(const BattleContext &bc) {
    // This first adapter has no hidden-intent or known-draw-prefix belief state.
    if (bc.player.hasRelic<R::RUNIC_DOME>() || bc.player.hasRelic<R::FROZEN_EYE>())
        throw std::invalid_argument("small playout v1 does not support Runic Dome/Frozen Eye beliefs");
    SmallFeatures f;
    auto &s = f.state;
    s[0] = float(bc.player.curHp) / 200; s[1] = float(bc.player.maxHp) / 200;
    s[2] = float(bc.player.block) / 100; s[3] = float(bc.player.energy) / 10;
    s[4] = float(bc.turn) / 20; s[5] = float(bc.potionCount) / 5;
    s[6] = float(bc.cards.cardsInHand) / 10;
    s[7] = float(bc.cards.drawPile.size()) / 40;
    s[8] = float(bc.cards.discardPile.size()) / 40;
    s[9] = float(bc.cards.exhaustPile.size()) / 40;
    s[10] = float(bc.player.strength) / 20; s[11] = float(bc.player.dexterity) / 20;
    s[12] = float(bc.player.getStatus<PS::WEAK>()) / 5;
    s[13] = float(bc.player.getStatus<PS::VULNERABLE>()) / 5;
    s[14] = float(bc.player.getStatus<PS::FRAIL>()) / 5;
    s[15] = float(bc.player.artifact) / 5;
    for (int i = 0; i < 5; ++i) {
        const auto &m = bc.monsters.arr[i];
        s[16 + 3*i] = float(std::max(0, m.curHp)) / 200;
        s[17 + 3*i] = float(m.block) / 100;
        s[18 + 3*i] = incoming(bc, m) / 50;
    }
    s[31] = bc.inputState == InputState::CARD_SELECT ? 1.f : 0.f;
    const auto actions = Action::enumerateCombatActions(bc);
    const auto choices = bc.inputState == InputState::CARD_SELECT ? combatChoices(bc) : CombatChoices{};
    for (const auto &action : actions) {
        SmallActionFeatures a; a.bits = action.bits;
        const auto kind = action.getActionType();
        a.x[static_cast<int>(kind)] = 1;
        a.x[13] = float(choices.zone) / 7;
        if (bc.inputState == InputState::CARD_SELECT) {
            a.x[14] = float(bc.cardSelectInfo.cardSelectTask) / 64;
            a.x[23] = float(bc.cardSelectInfo.selectedCount) / 10;
        }
        auto append = [&](const CardInstance &c) {
            addCard(a, c);
            a.x[9] += c.freeToPlayOnce;
            a.x[10] += c.doesExhaust(); a.x[11] += c.retain;
            a.x[12] += c.isEthereal();
        };
        if (kind == ActionType::CARD) append(bc.cards.hand[action.getSourceIdx()]);
        else if (kind == ActionType::POTION) {
            const int p = int(bc.potions[action.getSourceIdx()]);
            if (p < POTION_FIRST || p >= POTION_FIRST + POTION_KEYS)
                throw std::logic_error("invalid compact potion");
            a.keys.push_back(3 + CARD_KEYS + p - POTION_FIRST);
        } else if (kind == ActionType::SINGLE_CARD_SELECT) {
            if (bc.cardSelectInfo.cardSelectTask == CardSelectTask::CODEX && action.getSelectIdx() == 3)
                a.keys.push_back(1);
            else append(combatChoiceCard(bc, choices.zone, action.getSelectIdx()));
        } else if (kind == ActionType::MULTI_CARD_SELECT) {
            for (int index : action.getSelectedIdxs()) append(bc.cards.hand[index]);
            a.x[15] = float(a.keys.size()) / 10;
            if (a.keys.empty()) a.keys.push_back(2);
        } else a.keys.push_back(0);
        const bool targeted =
            (kind == ActionType::CARD && bc.cards.hand[action.getSourceIdx()].requiresTarget()) ||
            (kind == ActionType::POTION && potionRequiresTarget(bc.potions[action.getSourceIdx()]));
        if (targeted) {
            const int target = action.getTargetIdx();
            if (target >= 0 && target < 5) {
                const auto &m = bc.monsters.arr[target];
                a.x[16] = float(m.curHp) / 200; a.x[17] = float(m.maxHp) / 200;
                a.x[18] = float(m.block) / 100; a.x[19] = float(m.strength) / 20;
                a.x[20] = float(m.vulnerable) / 5; a.x[21] = float(m.weak) / 5;
                a.x[22] = float(target + 1) / 5;
            }
        }
        f.actions.push_back(std::move(a));
    }
    return f;
}

SmallPlayoutNet::SmallPlayoutNet(std::vector<float> weights) : weights_(std::move(weights)) {
    if (weights_.size() != std::size_t(4984 + 33 * smallKeyCount()) ||
        !std::all_of(weights_.begin(), weights_.end(), [](float x){ return std::isfinite(x); }))
        throw std::invalid_argument("compact policy parameter shape/finite contract");
}
std::vector<float> SmallPlayoutNet::logits(const SmallFeatures &f) const {
    const float *w = weights_.data();
    float h[64], z[32];
    for (int i=0; i<64; ++i) {
        float v=w[2048+i];
        for (int j=0; j<32; ++j) v+=w[i*32+j]*f.state[j];
        h[i]=std::max(0.f,v);
    }
    w += 2112;
    for (int i=0; i<32; ++i) {
        float v=w[2048+i];
        for (int j=0; j<64; ++j) v+=w[i*64+j]*h[j];
        z[i]=v;
    }
    const float *emb=weights_.data()+4192, *bias=emb+32*smallKeyCount();
    const float *proj=bias+smallKeyCount(), *lin=proj+768;
    std::vector<float> scores; scores.reserve(f.actions.size());
    for (const auto &a:f.actions) {
        float v=0;
        for (int k:a.keys) v+=bias[k]/float(a.keys.size());
        for (int j=0;j<24;++j) v+=lin[j]*a.x[j];
        for (int i=0;i<32;++i) {
            float e=0;
            for (int k:a.keys) e+=emb[k*32+i]/float(a.keys.size());
            for (int j=0;j<24;++j) e+=proj[i*24+j]*a.x[j];
            v+=z[i]*e;
        }
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite compact policy score");
        scores.push_back(v);
    }
    return scores;
}

double smallManualValue(const BattleContext &bc) {
    double removed=0, maximum=0, loot=0;
    for (const auto &m:bc.monsters.arr) {
        maximum+=std::max(0,m.maxHp); removed+=std::max(0,m.maxHp-m.curHp);
        // Match exitBattle's unrecovered gold, not an escape intent on a killed thief.
        const bool escaped = m.curHp > 0 && (m.moveHistory[0] == MMID::LOOTER_ESCAPE ||
                                            m.moveHistory[0] == MMID::MUGGER_ESCAPE);
        if ((m.id==MonsterId::LOOTER || m.id==MonsterId::MUGGER) &&
            (escaped || bc.outcome==Outcome::PLAYER_ESCAPE))
            loot+=std::max(0,m.miscInfo);
    }
    if (bc.outcome==Outcome::PLAYER_LOSS) {
        double quality=.6*(maximum>0?removed/maximum:0)+
            .25*std::max(0,bc.player.curHp)/std::max(1,bc.player.maxHp)+.15*bc.turn/(bc.turn+8.);
        return -1.45+.4*std::clamp(quality,0.,.999);
    }
    const double hp=std::clamp((bc.player.curHp+4.*bc.potionCount-.01*bc.turn)/200.,0.,1.);
    const double penalty=std::clamp(loot/1000.,0.,.45);
    if (bc.outcome==Outcome::PLAYER_VICTORY) return 1.+.5*hp-penalty;
    if (bc.outcome==Outcome::PLAYER_ESCAPE) return .5+.49*hp-penalty;
    return std::clamp(std::tanh(BattleScumSearcher2::evaluateEndState(bc)/20.)-penalty,-.999,.999);
}

namespace {
struct Node;
struct OutcomeNode {
    BattleContext state;
    std::uint64_t key;
    int samples=1;
    std::unique_ptr<Node> node;
};
struct Edge {
    Action action;
    std::uint64_t semantic;
    int visits=0;
    double sum=0;
    std::vector<OutcomeNode> outcomes;
};
struct Node { int visits=0; std::vector<Edge> edges; };
void expand(Node &node,const BattleContext &s,SmallSearchStats &stats) {
    for (auto a:Action::enumerateCombatActions(s))
        node.edges.push_back({a,NoslCombatTrainer::actionKey(s,a),0,0,{}});
    if(node.edges.empty()) throw std::runtime_error("nonterminal has no legal compact action");
    ++stats.nodes;
}
Action resolve(const Edge &e,const BattleContext &s) {
    if(e.action.isValidAction(s) && NoslCombatTrainer::actionKey(s,e.action)==e.semantic) return e.action;
    for(auto a:Action::enumerateCombatActions(s))
        if(NoslCombatTrainer::actionKey(s,a)==e.semantic) return a;
    throw std::logic_error("compact search action cannot be remapped");
}
double playout(BattleContext &s,const SmallPlayoutNet &net,const SmallSearchConfig &cfg,
               std::mt19937_64 &rng,SmallSearchStats &stats) {
    int steps=0;
    while(s.outcome==Outcome::UNDECIDED && steps<cfg.playoutDepth) {
        auto f=smallFeatures(s); auto logits=net.logits(f);
        if(logits.empty()) throw std::runtime_error("nonterminal playout without actions");
        ++stats.policyCalls; stats.actionScores+=logits.size();
        const float top=*std::max_element(logits.begin(),logits.end());
        std::vector<double> weights; weights.reserve(logits.size());
        for(float l:logits) weights.push_back(std::exp((l-top)/cfg.temperature));
        std::discrete_distribution<int> pick(weights.begin(),weights.end());
        Action(f.actions[pick(rng)].bits).execute(s);
        ++steps; ++stats.playoutTransitions;
    }
    ++stats.playoutLengths[steps];
    return smallManualValue(s);
}
}

SmallSearchResult smallSearch(const BattleContext &root,const SmallPlayoutNet &net,
                             const SmallSearchConfig &cfg,std::uint64_t nonce) {
    if(cfg.simulations<1 || cfg.playoutDepth<1 || cfg.playoutDepth>200 || cfg.treeDepth<1 ||
       cfg.treeDepth>1024 || !std::isfinite(cfg.exploration) || cfg.exploration<0 ||
       !std::isfinite(cfg.temperature) || cfg.temperature<=0)
        throw std::invalid_argument("invalid compact search budget");
    if(root.outcome!=Outcome::UNDECIDED) throw std::invalid_argument("search requires live combat");
    const auto started=std::chrono::steady_clock::now();
    SmallSearchResult result; result.features=smallFeatures(root);
    Node tree; expand(tree,root,result.stats);
    std::mt19937_64 rng(noslMix(nonce));
    for(int sim=0;sim<cfg.simulations;++sim) {
        BattleContext state(root);
#ifdef STS_ENABLE_RULE_TRACE
        state.atomicStepObserver = {};
#endif
        noslRedeterminize(state,noslMix(nonce+sim));
        Node *node=&tree; std::vector<Node*> nodes{node}; std::vector<Edge*> edges;
        double value=0; bool newLeaf=false; int depth=0;
        while(state.outcome==Outcome::UNDECIDED && depth<cfg.treeDepth) {
            std::vector<int> unvisited;
            for(int i=0;i<int(node->edges.size());++i) if(!node->edges[i].visits) unvisited.push_back(i);
            int chosen=0;
            if(!unvisited.empty()) chosen=unvisited[std::uniform_int_distribution<int>(0,int(unvisited.size())-1)(rng)];
            else {
                double best=-1e100;
                for(int i=0;i<int(node->edges.size());++i) {
                    const auto &e=node->edges[i];
                    const double score=e.sum/e.visits+cfg.exploration*std::sqrt(std::log(double(node->visits)+1)/e.visits);
                    if(score>best){best=score;chosen=i;}
                }
            }
            auto &edge=node->edges[chosen]; edges.push_back(&edge);
            const bool end=edge.action.getActionType()==ActionType::END_TURN;
            const int cap=int(std::ceil((end?23.148:1.)*std::pow(edge.visits+1.,end?.436:.5)));
            OutcomeNode *outcome=nullptr;
            if(int(edge.outcomes.size())<cap) {
                resolve(edge,state).execute(state); ++result.stats.treeTransitions;
                const auto key=NoslCombatTrainer::stateKey(state);
                for(auto &o:edge.outcomes) if(o.key==key){outcome=&o;++o.samples;break;}
                if(!outcome){
                    edge.outcomes.push_back({state,key,1,std::make_unique<Node>()});
                    outcome=&edge.outcomes.back(); newLeaf=true;
                }
            } else {
                std::vector<int> counts; for(const auto &o:edge.outcomes) counts.push_back(o.samples);
                std::discrete_distribution<int> choose(counts.begin(),counts.end());
                outcome=&edge.outcomes[choose(rng)]; state=outcome->state;
                noslRedeterminize(state,rng()); ++result.stats.chanceReplays;
            }
            node=outcome->node.get(); nodes.push_back(node); ++depth;
            if(state.outcome!=Outcome::UNDECIDED) break;
            if(newLeaf){expand(*node,state,result.stats);break;}
        }
        if(state.outcome!=Outcome::UNDECIDED) value=smallManualValue(state);
        else if(depth>=cfg.treeDepth){++result.stats.treeCuts;value=smallManualValue(state);}
        else value=playout(state,net,cfg,rng,result.stats);
        if(state.outcome==Outcome::PLAYER_VICTORY) ++result.stats.terminalWins;
        else if(state.outcome==Outcome::PLAYER_LOSS) ++result.stats.terminalLosses;
        else if(state.outcome==Outcome::PLAYER_ESCAPE) ++result.stats.terminalEscapes;
        else if(depth<cfg.treeDepth) ++result.stats.depthCuts;
        for(auto *n:nodes) ++n->visits;
        for(auto *e:edges){++e->visits;e->sum+=value;}
        ++result.stats.simulations;
    }
    int best=0;
    for(int i=0;i<int(tree.edges.size());++i){
        const auto &e=tree.edges[i]; result.visits.push_back(e.visits);
        result.values.push_back(e.visits?e.sum/e.visits:0.);
        if(e.visits>tree.edges[best].visits) best=i;
    }
    result.action=tree.edges[best].action.bits;
    result.stats.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return result;
}
}
