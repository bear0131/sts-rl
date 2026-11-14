#include <unordered_map>
#include <immintrin.h>
#include "sim/search/SmallPlayout.h"
#include "sim/search/PublicPlayoutAdapter.h"
#include "sim/search/CombatChoices.h"
#include "sim/search/Nosl.h"
#include "sim/search/NoslCombatTrainer.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace sts::search {
namespace {
constexpr int CARD_KEYS = 2 * static_cast<int>(CardId::ZAP);
constexpr int POTION_FIRST = static_cast<int>(Potion::AMBROSIA);
constexpr int POTION_KEYS = static_cast<int>(Potion::WEAK_POTION) - POTION_FIRST + 1;
std::atomic<std::uint64_t> nextSnapshotId{1};
constexpr int ACTION_PREFIX = 16;
struct ActionPrefixEntry {
    std::uint64_t snapshot = 0;
    int key = -1;
    std::array<float,ACTION_PREFIX> input{};
    std::array<float,SMALL_EMBED> embedding{};
    float bias = 0;
};
ActionPrefixEntry &prefixEntry(const SmallActionFeatures &action) {
    thread_local std::array<ActionPrefixEntry,1024> entries{};
    std::uint64_t hash = action.keys.front();
    // Cheap index only; every input byte is compared before a hit is accepted.
    for (const int j : {5,6,8,13,14,15}) {
        std::uint32_t bits;
        std::memcpy(&bits,&action.x[j],sizeof(bits));
        hash = (hash << 7) ^ (hash >> 3) ^ bits;
    }
    return entries[noslMix(hash) & (entries.size()-1)];
}
struct FullActionProjection {
    std::uint64_t snapshot = 0;
    int key = -1;
    std::array<float,SMALL_ACTION> input{};
    std::array<float,SMALL_EMBED> projected{};
    float bias = 0;
};

FullActionProjection &projectionEntry(const SmallActionFeatures &action) {
    thread_local std::array<FullActionProjection,2048> entries{};
    std::uint64_t hint = static_cast<std::uint32_t>(action.keys.front());
    // These fields select a bucket. Compare every action feature before reusing it.
    for (int j : {5,16,17,18,22}) {
        std::uint32_t word;
        std::memcpy(&word, &action.x[j], sizeof(word));
        hint = (hint << 5) ^ (hint >> 2) ^ word;
    }
    return entries[noslMix(hint) & (entries.size()-1)];
}

int cardKey(const CardInstance &c) {
    if (c.id == CardId::INVALID) throw std::logic_error("invalid compact card");
    return 3 + 2 * (static_cast<int>(c.id) - 1) + int(c.getUpgradeCount() > 0);
}
float incoming(const BattleContext &bc, const Monster &m) {
    if (!m.isTargetable() || bc.player.hasRelic<R::RUNIC_DOME>()) return 0;
    const auto damage = m.getMoveBaseDamage(bc);
    return float(m.getIntentDamage(bc, damage.damage) * damage.attackCount);
}
void addCard(SmallActionFeatures &f, const CardInstance &c) {
    f.addKey(cardKey(c));
    f.x[5] += float(c.costForTurn) / 5;
    f.x[6] += float(c.getUpgradeCount()) / 10;
    f.x[7] += float(c.getType()) / 5;
    f.x[8] += float(c.specialData) / 100;
}
}
int smallKeyCount() { return 3 + CARD_KEYS + POTION_KEYS; }

bool smallTurnCardLimitReached(const BattleContext &bc) {
    return bc.outcome==Outcome::UNDECIDED && bc.inputState==InputState::PLAYER_NORMAL &&
        bc.player.cardsPlayedThisTurn >= SMALL_MAX_CARDS_PER_TURN;
}
namespace {
std::vector<Action> budgetActions(const BattleContext &bc) {
    auto actions=completePolicyCombatActions(bc, CombatChoiceContract::NativeHandles);
    if(!smallTurnCardLimitReached(bc)) return actions;
    for(auto action:actions) if(action.getActionType()==ActionType::END_TURN) return {action};
    throw std::logic_error("turn card budget reached without a legal End Turn");
}
}

namespace {
void fillSmallState(const BattleContext &bc,SmallFeatures &f){
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
}
void featuresForActions(const BattleContext &bc, const std::vector<Action> &actions, SmallFeatures &f) {
    // Every state field is assigned below; retain only the actions' capacity.
    f.actions.clear();
    fillSmallState(bc,f);
    f.actions.reserve(actions.size());
    const auto choices = bc.inputState == InputState::CARD_SELECT
        ? combatChoices(bc, CombatChoiceContract::NativeHandles) : CombatChoices{};
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
            a.addKey(3 + CARD_KEYS + p - POTION_FIRST);
        } else if (kind == ActionType::SINGLE_CARD_SELECT) {
            if (bc.cardSelectInfo.cardSelectTask == CardSelectTask::CODEX && action.getSelectIdx() == 3)
                a.addKey(1);
            else append(combatChoiceCard(bc, choices.zone, action.getSelectIdx()));
        } else if (kind == ActionType::MULTI_CARD_SELECT) {
            for (int index : action.getSelectedIdxs()) append(bc.cards.hand[index]);
            a.x[15] = float(a.keyCount) / 10;
            if (a.keyCount==0) a.addKey(2);
        } else a.addKey(0);
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
}
}

SmallFeatures smallFeatures(const BattleContext &bc) {
    SmallFeatures features;
    featuresForActions(bc, budgetActions(bc), features);
    return features;
}

SmallPlayoutNet::SmallPlayoutNet(std::vector<float> weights)
    : snapshotId_(nextSnapshotId.fetch_add(1,std::memory_order_relaxed)), weights_(std::move(weights)) {
    if (weights_.size() != std::size_t(4984 + 33 * smallKeyCount()) ||
        !std::all_of(weights_.begin(), weights_.end(), [](float x){ return std::isfinite(x); }))
        throw std::invalid_argument("compact policy parameter shape/finite contract");
    // Immutable inference layout: independent output channels are contiguous.
    // Each output still accumulates inputs in the exported parameter order.
    for (int j=0;j<32;++j) for (int i=0;i<64;++i)
        stateColumns_[j*64+i]=weights_[i*32+j];
    for (int j=0;j<64;++j) for (int i=0;i<32;++i)
        hiddenColumns_[j*32+i]=weights_[2112+i*64+j];
    const float *projection=weights_.data()+4192+33*smallKeyCount();
    for (int j=0;j<SMALL_ACTION;++j) for (int i=0;i<SMALL_EMBED;++i)
        projectionColumns_[j*SMALL_EMBED+i]=projection[i*SMALL_ACTION+j];
}
std::vector<float> SmallPlayoutNet::logits(const SmallFeatures &f) const {
    std::vector<float> scores;
    logits(f,scores);
    return scores;
}
void SmallPlayoutNet::logits(const SmallFeatures &f, std::vector<float> &scores, const float *shift) const {
    float h[64], z[32];
    std::copy_n(weights_.data()+2048,64,h);
    for (int j=0;j<32;++j) {
        const float x=f.state[j], *column=stateColumns_.data()+j*64;
        if (x==0.f) continue;
        for (int i=0;i<64;++i) h[i]+=column[i]*x;
    }
    if(shift) for(int i=0;i<64;++i) h[i]+=shift[i];
    for (int i=0;i<64;++i) h[i]=std::max(0.f,h[i]);
    std::copy_n(weights_.data()+4160,32,z);
    for (int j=0;j<64;++j) {
        const float x=h[j], *column=hiddenColumns_.data()+j*32;
        if (x==0.f) continue;
        for (int i=0;i<32;++i) z[i]+=column[i]*x;
    }
    const float *emb=weights_.data()+4192, *bias=emb+32*smallKeyCount();
    const float *lin=bias+smallKeyCount()+SMALL_ACTION*SMALL_EMBED;
    scores.clear(); scores.reserve(f.actions.size());
    for (const auto &a:f.actions) {
        auto *entry = a.keyCount == 1 ? &projectionEntry(a) : nullptr;
        if (entry && entry->snapshot == snapshotId_ && entry->key == a.keys.front() &&
                std::memcmp(entry->input.data(), a.x.data(), SMALL_ACTION*sizeof(float)) == 0) {
            // The action projection is independent of state. Apply this state's z in the original order.
            float v = entry->bias;
            for (int i=0; i<SMALL_EMBED; ++i) v += z[i]*entry->projected[i];
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite compact policy score");
            scores.push_back(v);
            continue;
        }
        // Native action features are sparse. Retain the original column order:
        // factoring z through the projection would change float cancellation.
        int columns[SMALL_ACTION], count=0;
        for (int j=0;j<SMALL_ACTION;++j) if (a.x[j]!=0.f) columns[count++]=j;
        const bool singleKey=a.keyCount==1;
        int prefix=0;
        while (prefix<count && columns[prefix]<ACTION_PREFIX) ++prefix;
        auto *cached=singleKey ? &prefixEntry(a) : nullptr;
        float v=0;
        float e[SMALL_EMBED]{};
        if (cached && cached->snapshot==snapshotId_ && cached->key==a.keys.front() &&
                std::memcmp(cached->input.data(),a.x.data(),ACTION_PREFIX*sizeof(float))==0) {
            v=cached->bias;
            std::copy(cached->embedding.begin(),cached->embedding.end(),e);
        } else {
            if (singleKey) v+=bias[a.keys.front()];
            else for (int j=0;j<a.keyCount;++j) v+=bias[a.keys[j]]/float(a.keyCount);
            for (int col=0;col<prefix;++col) { const int j=columns[col]; v+=lin[j]*a.x[j]; }
            if (singleKey) {
                const float *embedding=emb+a.keys.front()*SMALL_EMBED;
                for (int i=0;i<SMALL_EMBED;++i) e[i]+=embedding[i];
            } else for (int j=0;j<a.keyCount;++j) {
                const float *embedding=emb+a.keys[j]*SMALL_EMBED;
                for (int i=0;i<SMALL_EMBED;++i) e[i]+=embedding[i]/float(a.keyCount);
            }
            for (int col=0;col<prefix;++col) {
                const int j=columns[col];
                const float x=a.x[j], *column=projectionColumns_.data()+j*SMALL_EMBED;
                for (int i=0;i<SMALL_EMBED;++i) e[i]+=column[i]*x;
            }
            if (cached) {
                cached->snapshot=snapshotId_; cached->key=a.keys.front(); cached->bias=v;
                std::copy_n(a.x.data(),ACTION_PREFIX,cached->input.data());
                std::copy_n(e,SMALL_EMBED,cached->embedding.data());
            }
        }
        // Vectorize independent outputs, retaining each output's input order.
        for (int col=prefix;col<count;++col) {
            const int j=columns[col];
            v+=lin[j]*a.x[j];
            const float x=a.x[j], *column=projectionColumns_.data()+j*SMALL_EMBED;
            for (int i=0;i<SMALL_EMBED;++i) e[i]+=column[i]*x;
        }
        if (entry) {
            entry->snapshot = snapshotId_;
            entry->key = a.keys.front();
            entry->input = a.x;
            std::copy_n(e, SMALL_EMBED, entry->projected.begin());
            entry->bias = v;
        }
        for (int i=0;i<SMALL_EMBED;++i) v+=z[i]*e[i];
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite compact policy score");
        scores.push_back(v);
    }
}

double smallTerminalValue(const BattleContext &bc, bool boss) {
    if (bc.outcome==Outcome::UNDECIDED)
        throw std::invalid_argument("unfinished battle has no terminal value");
    if (boss) return bc.outcome==Outcome::PLAYER_VICTORY ? 1. : 0.;
    return std::max(0,bc.player.curHp);
}

namespace {
void probabilities(const std::vector<float> &logits, double temperature, std::vector<double> &p, double mix=1.) {
    const auto top=*std::max_element(logits.begin(),logits.end());
    p.clear(); p.reserve(logits.size()); double sum=0;
    for(auto l:logits){p.push_back(std::exp((l-top)/temperature));sum+=p.back();}
    for(auto &v:p) v=mix*v/sum+(1.-mix)/p.size();
}
int drawPlayout(const std::vector<float> &logits, double temperature,
                std::vector<double> &weights, std::mt19937_64 &rng) {
    probabilities(logits,temperature,weights);
#ifdef __GLIBCXX__
    // Match libstdc++'s second normalization and lower_bound, but stop
    // forming cumulative probabilities once the draw is resolved.
    if (weights.size()==1) return 0;
    double total=0.;
    for (double weight:weights) total+=weight;
    const double draw=std::generate_canonical<double,std::numeric_limits<double>::digits>(rng);
    double cumulative=0.;
    for (std::size_t i=0;i+1<weights.size();++i) {
        cumulative+=weights[i]/total;
        if (draw<=cumulative) return static_cast<int>(i);
    }
    return static_cast<int>(weights.size()-1);
#else
    return std::discrete_distribution<int>(weights.begin(),weights.end())(rng);
#endif
}
std::uint64_t beliefKey(const BattleContext &s) {
    auto k=NoslCombatTrainer::streamingStateKey(s);
    const int known=s.player.hasRelic<R::FROZEN_EYE>() ? int(s.cards.drawPile.size()) :
        std::clamp(s.cards.knownDrawTop,0,int(s.cards.drawPile.size()));
    k=noslMix(k^known);
    for(int i=0;i<known;++i) k=noslMix(k^noslMix(s.cards.drawPile[s.cards.drawPile.size()-1-i].getUniqueId()+1));
    return k;
}
struct OutcomeNode;
struct Edge {
    Action action;
    std::uint64_t semantic;
    int visits=0;
    double sum=0;
    std::vector<std::unique_ptr<OutcomeNode>> outcomes;
    double prior=0;
    int terminalVisits=0;
};
struct Node { int visits=0, unvisited=0; std::vector<Edge> edges; };

// Internal nodes spend most of their lifetime at small visit counts. Only
// cache those rows; uncommon large pairs use exactly the same formula.
const double *explorationRow(int visits) {
    constexpr int limit=128;
    if (visits>limit) return nullptr;
    static const auto rows=[] {
        std::array<std::array<double,limit+1>,limit+1> table{};
        for (int n=1;n<=limit;++n)
            for (int k=1;k<=n;++k)
                table[n][k]=std::sqrt(std::log(double(n)+1)/k);
        return table;
    }();
    return rows[visits].data();
}
struct OutcomeNode {
    BattleContext state;
    std::uint64_t key;
    bool keyReady;
    int samples=1;
    Node node;

    // BattleContext has no move constructor. Keep it at a stable address so
    // growing an edge's outcomes never deep-copies already sampled states.
    OutcomeNode(const BattleContext &source, std::uint64_t belief, bool hasKey)
        : state(source), key(belief), keyReady(hasKey) {}

    std::uint64_t belief() {
        if (!keyReady) { key=beliefKey(state); keyReady=true; }
        return key;
    }
};
void expand(Node &node,const BattleContext &s,SmallSearchStats &stats,
            const SmallSearchConfig &cfg,const SmallPlayoutNet *prior) {
    for (auto a:budgetActions(s))
        node.edges.push_back({a,NoslCombatTrainer::strongActionKey(s,a),0,0,{}});
    node.unvisited=static_cast<int>(node.edges.size());
    if(node.edges.empty()) throw std::runtime_error("nonterminal has no legal compact action");
    if(cfg.usePrior) {
        std::vector<double> p;
        probabilities(prior->logits(smallFeatures(s)),cfg.priorTemperature,p,cfg.priorMix);
        for(std::size_t i=0;i<p.size();++i) node.edges[i].prior=p[i];
        ++stats.priorCalls;
    }
    ++stats.nodes;
}
Action resolve(const Edge &e,const BattleContext &s) {
    if(e.action.isValidAction(s) && NoslCombatTrainer::strongActionKey(s,e.action)==e.semantic) return e.action;
    for(auto a:budgetActions(s))
        if(NoslCombatTrainer::strongActionKey(s,a)==e.semantic) return a;
    std::ostringstream detail; detail << s;
    throw std::logic_error("compact search action cannot be remapped: turn="+std::to_string(s.turn)+
        " task="+std::to_string(int(s.cardSelectInfo.cardSelectTask))+" action="+std::to_string(e.action.bits)+
        " input="+std::to_string(int(s.inputState))+" semantic="+std::to_string(e.semantic)+"\n"+detail.str());
}
double playout(BattleContext &s,const SmallPlayoutNet &net,const SmallSearchConfig &cfg,
               std::mt19937_64 &rng,SmallSearchStats &stats,SmallPlayoutTrace *trace=nullptr) {
    int steps=0, observedTurn=-1;
    PublicShift turnShift;
    // Search states own no live observer. Scratch stays local to the worker
    // until its pool exits; each call rewrites all features, scores and weights.
    thread_local SmallFeatures f;
    thread_local std::vector<Action> actions;
    thread_local std::vector<float> logits;
    thread_local std::vector<double> weights;
    while(s.outcome==Outcome::UNDECIDED && steps<cfg.playoutDepth) {
        stats.turnCardLimitSimulated += smallTurnCardLimitReached(s);
        if (s.inputState == InputState::PLAYER_NORMAL) {
            actions.clear();
            if (smallTurnCardLimitReached(s)) actions.emplace_back(ActionType::END_TURN);
            else Action::appendNormalActions(s, actions, false);
        } else {
            actions = budgetActions(s);
        }
        if(actions.size()==1 && !trace) {
            // A one-action categorical distribution is exactly {1}, regardless
            // of its logit. Preserve the library's one-element RNG behavior.
#ifdef __GLIBCXX__
            const auto bits=actions.front().bits;
#else
            std::discrete_distribution<int> pick({1.});
            const auto bits=actions[pick(rng)].bits;
#endif
            ++stats.forcedPolicySkips;
            Action(bits).execute(s);
            ++steps; ++stats.playoutTransitions;
            continue;
        }
        featuresForActions(s,actions,f);
        if(cfg.publicRoot && cfg.publicRoot->refresh==ContextRefresh::State) {
            ++stats.publicStateRefreshes;
            const auto shift=cfg.publicRoot->liveShift(s);
            net.logits(f,logits,shift.data());
        } else if(cfg.publicRoot && cfg.publicRoot->refresh==ContextRefresh::Turn) {
            // Refresh on entering each simulated turn or pending selection.
            // The first action of every playout gets its actual leaf context.
            if(s.turn!=observedTurn || s.inputState==InputState::CARD_SELECT) {
                ++stats.publicStateRefreshes;
                turnShift=cfg.publicRoot->liveShift(s);
                // Do not carry pending-choice context into the subsequent ordinary decision.
                observedTurn=s.inputState==InputState::CARD_SELECT ? -1 : s.turn;
            }
            net.logits(f,logits,turnShift.data());
        } else net.logits(f,logits,cfg.publicRoot ? cfg.publicRoot->shift.data() : nullptr);
        if(logits.empty()) throw std::runtime_error("nonterminal playout without actions");
        ++stats.policyCalls; stats.actionScores+=logits.size();
        const int choice=drawPlayout(logits,cfg.temperature,weights,rng);
        const auto bits=f.actions[choice].bits;
        if(trace){trace->features.push_back(std::move(f));trace->choices.push_back(choice);}
        Action(bits).execute(s);
        ++steps; ++stats.playoutTransitions;
    }
    ++stats.playoutLengths[steps];
    return s.outcome==Outcome::UNDECIDED ? std::numeric_limits<double>::quiet_NaN() :
        smallTerminalValue(s,cfg.boss);
}
}

void smallRedeterminize(BattleContext &bc,std::uint64_t nonce) {
    auto draw=bc.cards.drawPile;
    noslRedeterminize(bc,nonce);
    bc.cards.drawPile=std::move(draw);
    const int known=bc.player.hasRelic<R::FROZEN_EYE>() ? int(bc.cards.drawPile.size()) :
        std::clamp(bc.cards.knownDrawTop,0,int(bc.cards.drawPile.size()));
    auto end=bc.cards.drawPile.end()-known;
    std::sort(bc.cards.drawPile.begin(),end,[](const auto &a,const auto &b){return a.getUniqueId()<b.getUniqueId();});
    java::Collections::shuffle(bc.cards.drawPile.begin(),end,java::Random(noslMix(nonce^0x534d414c4cULL)));
}

SmallPlayoutTrace smallCollectPlayout(const BattleContext &root,const SmallPlayoutNet &net,
                                    const SmallSearchConfig &cfg,std::uint64_t nonce) {
    if(root.player.hasRelic<R::RUNIC_DOME>())
        throw std::invalid_argument("compact pretraining requires public intents");
    if(cfg.playoutDepth<1 || cfg.playoutDepth>200 || cfg.temperature<=0 || !std::isfinite(cfg.temperature))
        throw std::invalid_argument("invalid compact playout budget");
    BattleContext state(root);
#ifdef STS_ENABLE_RULE_TRACE
    state.atomicStepObserver={};
#endif
    smallRedeterminize(state,nonce);
    SmallPlayoutTrace trace; std::mt19937_64 rng(noslMix(nonce));
    trace.value=playout(state,net,cfg,rng,trace.stats,&trace);
    trace.outcome=int(state.outcome);
    if(state.outcome==Outcome::UNDECIDED) ++trace.stats.depthCuts;
    return trace;
}

int smallRootPick(const std::vector<int> &visits,const std::vector<double> &values,
                  std::uint64_t nonce) {
    if(visits.empty() || visits.size()!=values.size())
        throw std::invalid_argument("root statistics shape mismatch");
    double top=-1e100;
    std::vector<int> ties;
    for(int i=0;i<int(visits.size());++i) {
        if(visits[i]<0 || !std::isfinite(values[i]))
            throw std::invalid_argument("invalid root statistic");
        if(!visits[i]) continue;
        if(values[i]>top){top=values[i];ties.clear();}
        if(values[i]==top) ties.push_back(i);
    }
    if(ties.empty()) throw std::invalid_argument("root needs at least one visited action");
    std::mt19937_64 rng(noslMix(nonce));
    return ties[std::uniform_int_distribution<int>(0,int(ties.size())-1)(rng)];
}

SmallSearchResult smallSearch(const BattleContext &root,const SmallPlayoutNet &net,
                             const SmallSearchConfig &cfg,std::uint64_t nonce,const SmallPlayoutNet *prior) {
    if(cfg.pairedWorldlines) {
        if(prior) throw std::invalid_argument("paired worldlines cannot use a separate tree prior");
        return smallPairedRootSearch(root,net,cfg,nonce);
    }

    if(root.player.hasRelic<R::RUNIC_DOME>())
        throw std::invalid_argument("small search requires a public-intent belief; use the public-only Dome policy");
    if(cfg.simulations<1 || cfg.playoutDepth<1 || cfg.playoutDepth>200 || cfg.treeDepth<1 ||
       cfg.treeDepth>1024 || !std::isfinite(cfg.exploration) || cfg.exploration < -1 ||
       !std::isfinite(cfg.temperature) || cfg.temperature<=0 ||
       !std::isfinite(cfg.priorMix) || cfg.priorMix<0 || cfg.priorMix>1 ||
       !std::isfinite(cfg.priorTemperature) || cfg.priorTemperature<=0 || (cfg.usePrior && !prior))
        throw std::invalid_argument("invalid compact search budget");
    if(root.outcome!=Outcome::UNDECIDED) throw std::invalid_argument("search requires live combat");
    const auto started=std::chrono::steady_clock::now();
    SmallSearchResult result; result.features=smallFeatures(root);
    result.stats.turnCardLimitDecisions=smallTurnCardLimitReached(root);
    const double scale=cfg.boss ? 1. : std::max(1,root.player.maxHp);
    const double exploration=cfg.exploration>=0?cfg.exploration:(cfg.simulations<500?.8:cfg.simulations<1000?1.41:2.5);
    Node tree; expand(tree,root,result.stats,cfg,prior);
    std::mt19937_64 rng(noslMix(nonce));
    for(int sim=0;sim<cfg.simulations;++sim) {
        BattleContext state(root);
#ifdef STS_ENABLE_RULE_TRACE
        state.atomicStepObserver = {};
#endif
        smallRedeterminize(state,noslMix(nonce+sim));
        Node *node=&tree; std::vector<Node*> nodes{node}; std::vector<Edge*> edges;
        double value=0; bool newLeaf=false; int depth=0;
        while(state.outcome==Outcome::UNDECIDED && depth<cfg.treeDepth) {
            result.stats.turnCardLimitSimulated += smallTurnCardLimitReached(state);
            int chosen=0;
            if(!cfg.usePrior && node->unvisited) {
                int ordinal=std::uniform_int_distribution<int>(0,node->unvisited-1)(rng);
                while (node->edges[chosen].visits || ordinal-- > 0) ++chosen;
            } else {
                const double *row=cfg.usePrior ? nullptr : explorationRow(node->visits);
                const double parentLog=!row && !cfg.usePrior ? std::log(double(node->visits)+1) : 0.;
                const double parentRoot=cfg.usePrior ? std::sqrt(double(node->visits)+1) : 0.;
                double best=-1e100; int ties=0;
                for(int i=0;i<int(node->edges.size());++i) {
                    const auto &e=node->edges[i];
                    const double q=e.visits?e.sum/e.visits:0.;
                    const double score=q+exploration*(cfg.usePrior ?
                        e.prior*parentRoot/(1+e.visits) :
                        (row ? row[e.visits] : std::sqrt(parentLog/e.visits)));
                    if(score>best){best=score;chosen=i;ties=1;}
                    else if(score==best && std::uniform_int_distribution<int>(1,++ties)(rng)==1) chosen=i;
                }
            }
            auto &edge=node->edges[chosen]; edges.push_back(&edge);
            if(!edge.visits) --node->unvisited;
            // The same progressive-widening rule for every action type.
            const auto outcomes=static_cast<std::int64_t>(edge.outcomes.size());
            OutcomeNode *outcome=nullptr;
            if(outcomes*outcomes<=edge.visits) {
                resolve(edge,state).execute(state); ++result.stats.treeTransitions;
                // The first outcome needs no equality comparison. Hash its
                // immutable snapshot only if another outcome is generated.
                const bool needsKey=!edge.outcomes.empty();
                const auto key=needsKey ? beliefKey(state) : std::uint64_t{0};
                for(auto &o:edge.outcomes) if(o->belief()==key){
                    if(state.inputState==InputState::CARD_SELECT && o->state.cardSelectInfo.cards!=state.cardSelectInfo.cards) {
                        std::ostringstream d; d << "generated choice key collision: ";
                        for(auto c:o->state.cardSelectInfo.cards) d << int(c) << ',';
                        d << " vs ";for(auto c:state.cardSelectInfo.cards) d << int(c) << ',';
                        throw std::logic_error(d.str());
                    }
                    outcome=o.get();++o->samples;break;
                }
                if(!outcome){
                    edge.outcomes.push_back(std::make_unique<OutcomeNode>(state,key,needsKey));
                    outcome=edge.outcomes.back().get(); newLeaf=true;
                }
            } else {
                std::vector<int> counts; for(const auto &o:edge.outcomes) counts.push_back(o->samples);
                std::discrete_distribution<int> choose(counts.begin(),counts.end());
                outcome=edge.outcomes[choose(rng)].get(); state=outcome->state;
                smallRedeterminize(state,rng()); ++result.stats.chanceReplays;
            }
            node=&outcome->node; nodes.push_back(node); ++depth;
            if(state.outcome!=Outcome::UNDECIDED) break;
            if(newLeaf){expand(*node,state,result.stats,cfg,prior);break;}
        }
        if(state.outcome!=Outcome::UNDECIDED) value=smallTerminalValue(state,cfg.boss);
        else if(depth>=cfg.treeDepth){++result.stats.treeCuts;value=std::numeric_limits<double>::quiet_NaN();}
        else value=playout(state,net,cfg,rng,result.stats);
        if(state.outcome==Outcome::PLAYER_VICTORY) ++result.stats.terminalWins;
        else if(state.outcome==Outcome::PLAYER_LOSS) ++result.stats.terminalLosses;
        else if(state.outcome==Outcome::PLAYER_ESCAPE) ++result.stats.terminalEscapes;
        else if(depth<cfg.treeDepth) ++result.stats.depthCuts;
        for(auto *n:nodes) ++n->visits;
        // Censored simulations supply no terminal observation. The backed-up
        // statistic is a conservative lower bound for nonnegative returns,
        // sum(observed terminal returns)/attempts, not a completed-only mean
        // or an assertion that an unfinished simulation was a loss.
        for(auto *e:edges){
            ++e->visits;
            if(std::isfinite(value)){e->sum+=value/scale;++e->terminalVisits;}
        }
        ++result.stats.simulations;
    }
    for(int i=0;i<int(tree.edges.size());++i){
        const auto &e=tree.edges[i]; result.visits.push_back(e.visits);
        result.terminalVisits.push_back(e.terminalVisits);
        result.values.push_back(e.visits?e.sum/e.visits:0.);
    }
    const int best=smallRootPick(result.visits,result.values,nonce);
    result.action=tree.edges[best].action.bits;
    result.stats.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return result;
}
PairedWorldlineResult smallPairedWorldlines(const BattleContext &root, const SmallPlayoutNet &net,
        const SmallSearchConfig &cfg, const std::vector<std::uint32_t> &bits, int budget, std::uint64_t nonce, bool keepWorlds) {
    if (bits.empty() || budget<1 || root.outcome!=Outcome::UNDECIDED || cfg.publicRoot ||
        cfg.playoutDepth<1 || cfg.playoutDepth>200 || !std::isfinite(cfg.temperature) || cfg.temperature<=0 ||
        root.player.hasRelic<R::RUNIC_DOME>())
        throw std::invalid_argument("paired worldlines require an active compact-policy combat root and finite budget");
    const auto legal=budgetActions(root);
    std::vector<Edge> choices;
    for (auto id:bits) {
        const auto it=std::find_if(legal.begin(),legal.end(),[id](Action a){return a.bits==id;});
        if (it==legal.end() || std::any_of(choices.begin(),choices.end(),[id](const Edge &e){return e.action.bits==id;}))
            throw std::invalid_argument("invalid or duplicate paired root action");
        choices.push_back({*it,NoslCombatTrainer::strongActionKey(root,*it),0,0,{}});
    }
    PairedWorldlineResult out;
    out.sums.resize(bits.size());out.visits.resize(bits.size());out.terminalVisits.resize(bits.size());
    std::uint64_t worldIndex=0;
    const double scale=cfg.boss ? 1. : std::max(1,root.player.maxHp);
    for (int remaining=budget;remaining>0;) {
        const auto world=noslMix(nonce+worldIndex++);
        BattleContext shared(root);
#ifdef STS_ENABLE_RULE_TRACE
        shared.atomicStepObserver={};
#endif
        smallRedeterminize(shared,world);
        std::vector<double> *values=keepWorlds ? &out.values.emplace_back() : nullptr;
        const auto count=std::min(choices.size(),static_cast<std::size_t>(remaining));
        if(values) values->reserve(count);
        for (std::size_t i=0;i<count;++i) {
            BattleContext state(shared);
            resolve(choices[i],state).execute(state);
            std::mt19937_64 rng(noslMix(world^0x504149524544ULL));
            ++out.stats.simulations;
            ++out.stats.playoutTransitions; // the committed candidate action in this simulation
            if(state.outcome!=Outcome::UNDECIDED) ++out.stats.playoutLengths[0];
            const auto value=state.outcome==Outcome::UNDECIDED ?
                playout(state,net,cfg,rng,out.stats) : smallTerminalValue(state,cfg.boss);
            out.completed+=std::isfinite(value);
            const double normalized=std::isfinite(value) ? value/scale : 0.;
            if(values) values->push_back(normalized);
            out.sums[i]+=normalized; ++out.visits[i]; out.terminalVisits[i]+=std::isfinite(value);
            if(state.outcome==Outcome::PLAYER_VICTORY) ++out.stats.terminalWins;
            else if(state.outcome==Outcome::PLAYER_LOSS) ++out.stats.terminalLosses;
            else if(state.outcome==Outcome::PLAYER_ESCAPE) ++out.stats.terminalEscapes;
            else ++out.stats.depthCuts;
        }
        remaining-=static_cast<int>(count);
    }
    return out;
}

SmallSearchResult smallPairedRootSearch(const BattleContext &root,const SmallPlayoutNet &net,
                                       const SmallSearchConfig &cfg,std::uint64_t nonce) {
    if(cfg.usePrior) throw std::invalid_argument("paired root search requires a single compact policy");
    const auto started=std::chrono::steady_clock::now();
    SmallSearchResult result;
    result.features=smallFeatures(root);
    const auto n=result.features.actions.size();
    std::vector<std::size_t> order(n);
    for(std::size_t i=0;i<n;++i) order[i]=i;
    std::mt19937_64 rng(noslMix(nonce^0x5045524dULL));
    std::shuffle(order.begin(),order.end(),rng); // randomize which actions get the remaining units
    std::vector<std::uint32_t> actions;actions.reserve(n);
    for(auto i:order)actions.push_back(result.features.actions[i].bits);
    const auto panel=smallPairedWorldlines(root,net,cfg,actions,cfg.simulations,noslMix(nonce),false);
    result.stats=panel.stats;
    result.stats.turnCardLimitDecisions=smallTurnCardLimitReached(root);
    result.visits.resize(n);result.terminalVisits.resize(n);result.values.resize(n);
    double best=-std::numeric_limits<double>::infinity();
    for(std::size_t i=0;i<n;++i){
        const auto j=order[i];
        result.visits[j]=panel.visits[i];result.terminalVisits[j]=panel.terminalVisits[i];
        if(result.visits[j]){
            result.values[j]=panel.sums[i]/result.visits[j];
            best=std::max(best,result.values[j]);
        }
    }
    // The reference gate used half a percentage point of normalized terminal
    // HP, or of win probability at bosses. Break remaining policy ties fairly.
    const auto scores=net.logits(result.features);
    ++result.stats.policyCalls;result.stats.actionScores+=n;
    float top=-std::numeric_limits<float>::infinity();std::vector<std::size_t> tied;
    for(std::size_t i=0;i<n;++i){
        if(!result.visits[i] || result.values[i]<best-.005)continue;
        if(scores[i]>top){top=scores[i];tied.clear();}
        if(scores[i]==top)tied.push_back(i);
    }
    if(tied.empty())throw std::logic_error("paired search needs a scored legal action");
    const auto chosen=tied[std::uniform_int_distribution<std::size_t>(0,tied.size()-1)(rng)];
    result.action=result.features.actions[chosen].bits;
    result.stats.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return result;
}

}

#include "sim/search/WaveDiag.h"
namespace sts::search {
WaveRoot::WaveRoot(const GameContext &gc,const BattleContext &bc,SmallSearchConfig c,std::uint64_t n,bool paired):pairedAnchor(paired),root(bc),inventory(playoutInventory(gc)),deck(gc.deck.size()),cfg(c),nonce(n) {
    if((!pairedAnchor&&root.player.hasRelic<R::RUNIC_DOME>())||root.outcome!=Outcome::UNDECIDED||cfg.playoutDepth<1||cfg.playoutDepth>200)throw std::logic_error("invalid panel");
#ifdef STS_ENABLE_RULE_TRACE
    root.atomicStepObserver={};
#endif
    if(!pairedAnchor){
        auto rootObs=smallFeatures(root);for(auto &x:rootObs.actions){actions.emplace_back(x.bits);semantic.push_back(NoslCombatTrainer::strongActionKey(root,actions.back()));order.push_back(int(order.size()));}
        std::mt19937_64 shuffle(noslMix(nonce^0x5045524dULL));std::shuffle(order.begin(),order.end(),shuffle);
    }
    auto &draw=root.cards.drawPile;const int drawCount=int(draw.size()),known=root.player.hasRelic<R::FROZEN_EYE>()?drawCount:std::clamp(root.cards.knownDrawTop,0,drawCount);
    canonical.assign(draw.begin(),pairedAnchor?draw.end():draw.end()-known);
    std::sort(canonical.begin(),canonical.end(),[](const auto &a,const auto &b){return a.getUniqueId()<b.getUniqueId();});

}
void WaveRoot::configurePaired(const std::vector<std::uint32_t>&bits,int selected,
        const std::vector<std::array<std::uint64_t,5>>&words){
    if(!pairedAnchor||bits.size()<2||words.empty()||!actions.empty())throw std::logic_error("invalid paired root");
    prefix=selected;worlds=words;
    for(auto b:bits){Action a(b);if(!validAutoregressiveAction(a,root,prefix))throw std::logic_error("illegal paired root action");
        actions.push_back(a);semantic.push_back(int(a.getActionType())<5?NoslCombatTrainer::strongActionKey(root,a):0);order.push_back(order.size());}
    if(cfg.simulations!=int(actions.size()*worlds.size()))throw std::logic_error("incomplete paired world");
}
static bool executePaired(Action a,BattleContext&bc,int&prefix){
    if(a.getActionType()==kSelectNext){prefix|=1<<a.getSourceIdx();return false;}
    if(a.getActionType()==kSelectStop)Action(ActionType::MULTI_CARD_SELECT,prefix).execute(bc);else a.execute(bc);
    prefix=0;return true;
}
static double pairedDraw(std::array<std::uint64_t,4>&s){
    using U=unsigned __int128;
    const U multiplier=(U(0x2360ed051fc65da4ULL)<<64)|0x4385df649fccf645ULL;
    U state=(U(s[1])<<64)|s[0],inc=(U(s[3])<<64)|s[2];state=state*multiplier+inc;
    s[0]=std::uint64_t(state);s[1]=std::uint64_t(state>>64);
    auto x=s[0]^s[1];unsigned r=s[1]>>58;return double(((x>>r)|(x<<((-r)&63)))>>11)*0x1.0p-53;
}
void applyRootWorld(BattleContext &bc,const WaveRoot &r,std::uint64_t nonce){
    auto streamSeed=[](std::uint64_t v,std::uint64_t k){return noslMix(v^noslMix(k));};
    bc.aiRng = sts::Random(streamSeed(nonce, 1));
    bc.cardRandomRng = sts::Random(streamSeed(nonce, 2));
    bc.miscRng = sts::Random(streamSeed(nonce, 3));
    bc.monsterHpRng = sts::Random(streamSeed(nonce, 4));
    bc.potionRng = sts::Random(streamSeed(nonce, 5));
    bc.shuffleRng = sts::Random(streamSeed(nonce, 6));
    auto &draw=bc.cards.drawPile;
    std::copy(r.canonical.begin(),r.canonical.end(),draw.begin());
    java::Collections::shuffle(draw.begin(),draw.begin()+r.canonical.size(),java::Random(r.pairedAnchor?streamSeed(nonce,7):noslMix(nonce^0x534d414c4cULL)));
}
static std::shared_ptr<NeuralSearchBatch> reusePool(int n){
    static std::mutex mutex;static std::unordered_map<int,std::shared_ptr<NeuralSearchBatch>> pools;
    std::lock_guard<std::mutex> lock(mutex);auto &p=pools[n];if(!p)p=std::make_shared<NeuralSearchBatch>(n);return p;
}
WaveDiag::WaveDiag(std::vector<std::shared_ptr<WaveRoot>> v,int batch,int threads):poolOwner(reusePool(threads)),pool(*poolOwner){
    resetGroups(std::move(v),batch);
}
void WaveDiag::resetGroups(std::vector<std::shared_ptr<WaveRoot>> v,int batch){
    if(batch<1||batch>8192||v.empty())throw std::logic_error("invalid next group");
    if(awaiting||nextTask.load()<int(local.size()))throw std::logic_error("cannot reset running search");
    for(const auto &l:lanes)if(l.task>=0||l.ready)throw std::logic_error("live lane at reset");
    roots=std::move(v);lanes.resize(batch);nextTask.store(0);active.clear();pendingPicks.clear();
    workRoot.clear();local.clear();chosen.clear();worldOffsets.clear();starts.reset();
    depths.clear();forceN.clear();outcomes.clear();actionN.clear();values.clear();prefixN.clear();maxActionN.clear();pendingValues.clear();
    for(auto &l:lanes){
        l.lastRow=-1;l.sharedIndex=-1;l.drawIndex=0;l.chosenNext=-1;l.pubSec=l.encodeSec=0;l.dirtyTokens=l.allTokens=0;l.leafKind=0;l.prefix=0;
        l.forced=l.steps=l.scored=0;l.cached.reset();
        if(l.cardCache)for(auto &z:l.cardCache->zones)z.deck=-1;
    }
    counterDraw=std::getenv("WAVE_COUNTER_DRAW") && std::atoi(std::getenv("WAVE_COUNTER_DRAW"))==1;
    auto flag=std::getenv("SHARE_WORLD_MODE");shareMode=flag?std::atoi(flag):0;
    if(shareMode<0||shareMode>3)throw std::logic_error("invalid world sharing mode");
    if(counterDraw)shareMode=0;
    if(shareMode){int total=0;for(auto &r:roots){worldOffsets.push_back(total);int choices=int(r->order.size());if(!choices)throw std::logic_error("empty root");total+=(r->cfg.simulations+choices-1)/choices;}
        starts=std::make_unique<SharedStart[]>(total);
        for(std::size_t i=0;i<roots.size();++i){int a=roots[i]->order.size();for(int n=0;n<roots[i]->cfg.simulations;n+=a)starts[worldOffsets[i]+n/a].remaining=std::min(a,roots[i]->cfg.simulations-n);}
    }
    // Interleave roots so different combat lengths and tiers coexist throughout the stream.
    int mx=0;for(auto &r:roots)mx=std::max(mx,r->cfg.simulations);
    const int schedule=std::getenv("TAIL_ORDER")?std::atoi(std::getenv("TAIL_ORDER")):0;
    auto addTask=[&](int i,int j){workRoot.push_back(i);local.push_back(j);chosen.push_back(roots[i]->order[j%roots[i]->order.size()]);};
    if(schedule==2){
        std::vector<int> priority(roots.size());std::iota(priority.begin(),priority.end(),0);
        std::vector<int> hp(roots.size(),0);
        for(int i:priority)for(const auto &m:roots[i]->root.monsters.arr)if(m.id!=MonsterId::INVALID)hp[i]+=std::max(0,int(m.curHp));
        std::stable_sort(priority.begin(),priority.end(),[&](int a,int b){return hp[a]>hp[b];});
        for(int i:priority)for(int j=0;j<roots[i]->cfg.simulations;++j)addTask(i,j);
    }else if(schedule==1){
        for(int phase=0;phase<mx;++phase)for(int i=0;i<int(roots.size());++i){
            int n=roots[i]->cfg.simulations;
            for(int j=phase*n/mx;j<(phase+1)*n/mx;++j)addTask(i,j);
        }
    }else{
        for(int j=0;j<mx;++j)for(int i=0;i<int(roots.size());++i)if(j<roots[i]->cfg.simulations)addTask(i,j);
    }
    depths.resize(local.size());forceN.resize(local.size());outcomes.resize(local.size());prefixN.resize(local.size());maxActionN.resize(local.size());actionN.resize(local.size());values.resize(local.size(),std::numeric_limits<double>::quiet_NaN());
}
static unsigned waveHash(unsigned x){x=(x^(x>>16))*0x7feb352du;x=(x^(x>>15))*0x846ca68bu;return x^(x>>16);}
struct WTable {unsigned id=~0u;std::array<unsigned,33> code{};};
static const WTable &waveTable(int id,int z){
    thread_local std::array<WTable,4096> cache;
    unsigned a=id*15+z,k=(id/512+6144)*15+z;
    auto &e=cache[waveHash(a)&4095];if(e.id!=a){e.id=a;e.code[0]=waveHash(a*17);
    for(int j=0;j<16;++j){e.code[1+j]=waveHash(a*17+j+1);e.code[17+j]=waveHash(k*17+j+1);}}return e;
}
static void plusVal(std::array<float,128>&f,unsigned h,float v){f[h%127]+=h>>31?-v:v;}
static int normalFactBound(const BattleContext &bc){
    const auto &p=bc.player;int n=2+21+4+__builtin_popcountll(p.statusBits0)+__builtin_popcountll(p.statusBits1)+int(p.bombs.size());
    n+=__builtin_popcountll(p.relicBits0)+__builtin_popcountll(p.relicBits1)+bc.onUseRelicOrder.size()+bc.onEndTurnRelicOrder.size()+bc.onPostDrawRelicOrder.size()+p.endTurnPowerOrder.size();
    n+=bc.potionCapacity+5;
    for(int i=0;i<5;++i){const auto &m=bc.monsters.arr[i];if(m.id!=MonsterId::INVALID)n+=5+__builtin_popcountll(m.statusBits);}
    return n;
}
void WaveDiag::encode(WaveLane &l){
    auto &r=*roots[workRoot[l.task]];auto &e=l.entity;
    // Current compact observation is packed in the existing native batch
    // transport. Its GPU consumer owns token projection and query compaction.
    if(r.pairedAnchor)return;
    auto encodeStart=std::chrono::steady_clock::now();const bool plain=aosMode&&l.state->inputState==InputState::PLAYER_NORMAL;
    static const std::vector<SmallActionFeatures> noActions;
    if(plain){l.factBound=normalFactBound(*l.state);e.entities.clear();e.sources.clear();e.targets.clear();}
    else playoutEntities(*l.state,r.deck,l.obs.actions,e);
    auto pubEnd=std::chrono::steady_clock::now();l.pubSec+=std::chrono::duration<double>(pubEnd-encodeStart).count();int battleRows=e.entities.size();if(!aosMode)e.entities.insert(e.entities.end(),r.inventory.begin(),r.inventory.end());

    if(plain){
        auto &state=*l.state;l.rawExtra.clear();l.live.resize(33);auto m=l.obs.actions.size();l.src.resize(m);l.target.resize(m);
        std::uint64_t query=1;
        for(std::size_t i=0;i<m;++i){Action a(l.obs.actions[i].bits);int src=-1,tg=-1;
            if(a.getActionType()==ActionType::CARD){int j=a.getSourceIdx();src=18+j;if(state.cards.hand[j].requiresTarget())tg=28+a.getTargetIdx();}
            else if(a.getActionType()==ActionType::POTION){int j=a.getSourceIdx();src=13+j;if(potionRequiresTarget(state.potions[j]))tg=28+a.getTargetIdx();}
            else if(a.getActionType()!=ActionType::END_TURN)throw std::logic_error("unknown normal play candidate");
            l.src[i].fill(-1);l.src[i][0]=src;l.target[i]=tg;if(src>=0)query|=1ull<<src;if(tg>=0)query|=1ull<<tg;
        }
        int mapping[33];l.queries.clear();while(query){int j=__builtin_ctzll(query);query&=query-1;mapping[j]=l.queries.size();l.queries.push_back(j);}
        for(std::size_t i=0;i<m;++i){auto &src=l.src[i][0];auto &tg=l.target[i];if(src>=0)src=mapping[src];if(tg>=0)tg=mapping[tg];}
        l.encodeSec+=std::chrono::duration<double>(std::chrono::steady_clock::now()-pubEnd).count();return;
    }
    const auto n=e.entities.size();thread_local std::vector<int> owner,sourceSlot,count;thread_local std::vector<char> used;
    owner.resize(n);sourceSlot.assign(n,-1);used.assign(n,false);
    for(const auto &a:e.sources)for(auto i:a)if(i>=0)used.at(i)=true;
    int extra=0;constexpr int zoneSlot[]={1,2,18,6,7,8,9,10,12,11,5,3,4,13,28};
    for(std::size_t i=0;i<n;++i){
        auto &o=e.entities[i];int z=o.zone,s=zoneSlot[z];
        if(z==1&&o.id/512==1)s=1;
        if(z==2){s+=int(std::nearbyint(o.numbers[8]*40))-1;sourceSlot[i]=s;}
        if(z==13){s+=int(std::nearbyint(o.numbers[0]*5))-1;sourceSlot[i]=s;}
        if(z==14){s+=o.monster;if(o.id/512==5)sourceSlot[i]=s;}
        if(used[i]&&sourceSlot[i]<0)sourceSlot[i]=33+extra++;
        owner[i]=s;
    }

    const int t=33+extra;
    if(rawMode&&fullTokens){
        count.assign(t,0);l.symbols.assign(t,0);
        for(std::size_t i=0;i<n;++i){++count[owner[i]];int ss=sourceSlot[i];if(ss>=0){l.symbols[ss]=e.entities[i].id+1;if(ss>=33)++count[ss];}}
        l.rawCounts=count;
    }else if(fastCodec&&fullTokens&&!extra){if(!l.cached)l.cached=std::make_unique<WaveLane::TokenCache>();
        count.assign(33,0);l.symbols.assign(33,0);l.features.resize(33);
        std::array<bool,33> dirty{};
        // Compare the complete public row, not a hash. Unchanged slots can reuse the exact normalized token.
        // Empty slots retain their actual native slot and symbols.
        for(std::size_t i=0;i<n;++i){
            const auto &row=e.entities[i];int j=owner[i], k=count[j]++;auto &before=l.cached->priorRows[j];
            if(k>=int(before.size())){before.push_back(row);dirty[j]=true;}
            else if(std::memcmp(&before[k],&row,sizeof(PlayoutEntity))!=0){before[k]=row;dirty[j]=true;}
            if(sourceSlot[i]>=0)l.symbols[sourceSlot[i]]=row.id+1;
        }
        for(int j=0;j<33;++j){
            auto &before=l.cached->priorRows[j];if(before.size()!=std::size_t(count[j])){before.resize(count[j]);dirty[j]=true;}
            auto &f=l.cached->cachedTokens[j];if(count[j])++l.allTokens;
            if(dirty[j]){
                ++l.dirtyTokens;f.fill(0);
                for(auto &row:before)plusVal(f,waveTable(row.id,row.zone).code[0],1.f);
                for(int pass=0;pass<2;++pass)for(auto &row:before){
                    const auto &code=waveTable(row.id,row.zone).code;
                    // Explicit AVX2 nonzero masks avoid checking 16 sparse components one at a time.
                    const float *v=row.numbers.data();const auto z=_mm256_setzero_ps();
                    unsigned mask=unsigned(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(v),z,_CMP_NEQ_OQ))) |
                        (unsigned(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(v+8),z,_CMP_NEQ_OQ)))<<8);
                    while(mask){int k=__builtin_ctz(mask);mask&=mask-1;plusVal(f,code[1+pass*16+k],v[k]);}
                }
                if(count[j]>1){const float scale=std::sqrt(float(count[j]));for(auto &v:f)v/=scale;}
                f[127]=std::log1p(float(count[j]));
            }
        }
    }else{
l.features.assign(t,{});l.symbols.assign(t,0);count.assign(t,0);
    for(std::size_t i=0;i<n;++i){auto &o=e.entities[i];int s=owner[i],ss=sourceSlot[i];auto &code=waveTable(o.id,o.zone).code;
        plusVal(l.features[s],code[0],1);++count[s];
        if(ss>=0){l.symbols[ss]=o.id+1;if(ss>=33){plusVal(l.features[ss],code[0],1);++count[ss];}}
    }
    for(int pass=0;pass<2;++pass)for(std::size_t i=0;i<n;++i){
        auto &o=e.entities[i];int s=owner[i],ss=sourceSlot[i];auto &code=waveTable(o.id,o.zone).code;
        for(int j=0;j<16;++j)if(o.numbers[j]!=0){auto h=code[1+pass*16+j];plusVal(l.features[s],h,o.numbers[j]);if(ss>=33)plusVal(l.features[ss],h,o.numbers[j]);}
    }
    for(int i=0;i<t;++i)if(count[i]){auto &f=l.features[i];float scale=std::sqrt(float(count[i]));if(count[i]>1)for(auto &v:f)v/=scale;f[127]=std::log1p(float(count[i]));}
    }
    if(!fullTokens){l.ctx.resize(1552);int k=0;for(int i:{3,4,5,1,2,6,7,8,9,10,11,12}){std::copy_n(l.features[i].data(),128,l.ctx.data()+k);k+=128;}
    for(int i=0;i<5;++i){l.ctx[1536+i]=l.symbols[28+i]==2561;l.ctx[1541+i]=l.symbols[13+i]==4098;}}
    l.live.clear();l.live.push_back(0);l.queries.clear();l.queries.push_back(0);l.qindex.assign(t,-1);l.qindex[0]=0;
    thread_local std::vector<char> asked;asked.assign(t,false);asked[0]=true;
    l.src=e.sources;l.target=e.targets;
    for(auto &a:l.src)for(auto &i:a)if(i>=0){i=sourceSlot.at(i);asked.at(i)=true;}
    for(auto &i:l.target)if(i>=0){i+=28;asked.at(i)=true;}
    for(int i=(fullTokens?1:13);i<t;++i){
        const bool empty=(i>=28&&i<33&&l.symbols[i]==2561)||(i>=13&&i<18&&l.symbols[i]==4098);
        if(!aosMode&&!asked[i]&&((!fullTokens&&empty)||!count[i]))continue;
        if(asked[i]){l.qindex[i]=l.queries.size();l.queries.push_back(l.live.size());}
        l.live.push_back(i);
    }
    for(auto &a:l.src)for(auto &i:a)if(i>=0)i=l.qindex.at(i);
    for(auto &i:l.target)if(i>=0)i=l.qindex.at(i);
    if(rawMode&&fullTokens){
        thread_local std::vector<int> compact;compact.assign(t,-1);for(int j=0;j<int(l.live.size());++j)compact[l.live[j]]=j;
        l.rawOwner.resize(n);l.rawExtra.resize(n);
        for(std::size_t j=0;j<n;++j){l.rawOwner[j]=compact[owner[j]];l.rawExtra[j]=sourceSlot[j]>=33?compact[sourceSlot[j]]:-1;}
    }
    l.encodeSec+=std::chrono::duration<double>(std::chrono::steady_clock::now()-pubEnd).count();
}
void WaveDiag::prepareLane(std::size_t i){
        auto &l=lanes[i];l.ready=false;
        if(l.leafKind&&l.lastRow>=0){
            const double value=pendingValues.at(l.lastRow);if(!std::isfinite(value))throw std::logic_error("nonfinite paired leaf");
            values[l.task]=value;depths[l.task]=l.depth;forceN[l.task]=l.jobForced;
            outcomes[l.task]=l.leafKind;actionN[l.task]=l.jobActions;prefixN[l.task]=l.jobPrefixes;maxActionN[l.task]=l.jobMaxActions;
            l.task=-1;l.leafKind=0;l.lastRow=-1;
        }
        if(gpuChoice&&l.lastRow>=0)l.chosenNext=pendingPicks.at(l.lastRow);
        if(l.chosenNext>=0){
            int j=l.chosenNext; l.chosenNext=-1;l.lastRow=-1;
            if(j>=int(l.obs.actions.size()))throw std::logic_error("GPU chose illegal candidate");
            if(roots[workRoot[l.task]]->pairedAnchor)l.jobPrefixes+=!executePaired(Action(l.obs.actions[j].bits),*l.state,l.prefix);
            else Action(l.obs.actions[j].bits).execute(*l.state);
            ++l.steps;++l.depth;++l.scored;l.jobActions+=l.obs.actions.size();
        }thread_local std::vector<Action> acts;
        for(;;){
            if(l.task<0){
                int task=nextTask.fetch_add(1);if(task>=int(local.size()))return;
                l.task=task;if(resident&&l.cardCache)for(auto &z:l.cardCache->zones)z.deck=-1;auto &r=*roots[workRoot[task]];auto world=noslMix(noslMix(r.nonce)+local[task]/r.order.size());
                if(r.pairedAnchor){const auto&w=r.worlds.at(local[task]/r.order.size());world=w[0];std::copy_n(w.begin()+1,4,l.pcg.begin());}
                l.sharedIndex=-1;l.drawIndex=0;l.policySeed=noslMix(world^0x504149524544ULL);
                if(shareMode&&gpuChoice&&!r.pairedAnchor){
                    l.sharedIndex=worldOffsets[workRoot[task]]+local[task]/r.order.size();auto &entry=starts[l.sharedIndex];
                    std::call_once(entry.once,[&]{
                        if(shareMode&1){std::mt19937_64 stream(l.policySeed);for(auto &u:entry.uniforms)u=std::generate_canonical<double,std::numeric_limits<double>::digits>(stream);}
                        if(shareMode&2){entry.base=std::make_unique<BattleContext>(r.root);applyRootWorld(*entry.base,r,world);}
                    });
                    if(shareMode&2){if(l.state)*l.state=*entry.base;else l.state=WaveStatePtr(new BattleContext(*entry.base));if(entry.remaining.fetch_sub(1,std::memory_order_acq_rel)==1)entry.base.reset();}
                    else {if(l.state)*l.state=r.root;else l.state=WaveStatePtr(new BattleContext(r.root));applyRootWorld(*l.state,r,world);}
                }else {if(l.state)*l.state=r.root;else l.state=WaveStatePtr(new BattleContext(r.root));applyRootWorld(*l.state,r,world);}
                int a=chosen[task];Edge edge{r.actions[a],r.semantic[a],0,0,{}};l.prefix=r.prefix;l.jobPrefixes=0;l.jobMaxActions=0;
                if(r.pairedAnchor)l.jobPrefixes+=!executePaired(int(r.actions[a].getActionType())>=5?r.actions[a]:resolve(edge,*l.state),*l.state,l.prefix);
                else resolve(edge,*l.state).execute(*l.state);
                if(!r.pairedAnchor&&!counterDraw&&(l.sharedIndex<0||!(shareMode&1))){if(!l.rng)l.rng=std::make_unique<std::mt19937_64>();l.rng->seed(l.policySeed);}l.depth=r.pairedAnchor?1:0;l.jobForced=0;l.jobActions=0;++l.steps;
            }
            auto &s=*l.state;auto &r=*roots[workRoot[l.task]];
            if(s.outcome!=Outcome::UNDECIDED||l.depth>=r.cfg.playoutDepth){
                if(r.pairedAnchor){
                    l.leafKind=s.outcome==Outcome::UNDECIDED?4:int(s.outcome);
                    if(s.outcome==Outcome::PLAYER_LOSS){values[l.task]=0;depths[l.task]=l.depth;forceN[l.task]=l.jobForced;outcomes[l.task]=2;actionN[l.task]=l.jobActions;prefixN[l.task]=l.jobPrefixes;maxActionN[l.task]=l.jobMaxActions;l.task=-1;l.leafKind=0;continue;}
                    l.obs.actions.clear();l.ready=true;return;
                }
                if(s.outcome!=Outcome::UNDECIDED)values[l.task]=smallTerminalValue(s,r.cfg.boss)/(r.cfg.boss?1.:std::max(1,r.root.player.maxHp));
                depths[l.task]=l.depth;forceN[l.task]=l.jobForced;outcomes[l.task]=int(s.outcome);actionN[l.task]=l.jobActions;
                l.task=-1;continue;
            }
            acts.clear();
            if(r.pairedAnchor)acts=autoregressiveActions(s,l.prefix);
            else if(s.inputState==InputState::PLAYER_NORMAL){
                if(smallTurnCardLimitReached(s))acts.emplace_back(ActionType::END_TURN);
                else Action::appendNormalActions(s,acts,false);
            }else acts=budgetActions(s);
            l.jobMaxActions=std::max(l.jobMaxActions,int(acts.size()));
            if(acts.size()==1){if(r.pairedAnchor)l.jobPrefixes+=!executePaired(acts[0],s,l.prefix);else acts[0].execute(s);++l.steps;++l.depth;++l.forced;++l.jobForced;continue;}
            if(acts.empty())throw std::logic_error("empty live choices");
            if(r.pairedAnchor){std::stable_sort(acts.begin(),acts.end(),[&](Action a,Action b){return jointActionId(a,s)<jointActionId(b,s);});l.obs.actions.resize(acts.size());for(std::size_t j=0;j<acts.size();++j)l.obs.actions[j].bits=acts[j].bits;}
            else if(shortAtoms&&s.inputState==InputState::PLAYER_NORMAL){fillSmallState(s,l.obs);l.obs.actions.resize(acts.size());for(std::size_t j=0;j<acts.size();++j)l.obs.actions[j].bits=acts[j].bits;}
            else featuresForActions(s,acts,l.obs);
            encode(l);
            if(gpuChoice){
                if(r.pairedAnchor)l.drawNext=pairedDraw(l.pcg);
                else if(counterDraw)l.drawNext=double(noslMix(l.policySeed+0x9e3779b97f4a7c15ULL*std::uint64_t(l.drawIndex))>>11)*0x1.0p-53;
                else if(l.sharedIndex>=0&&(shareMode&1)&&l.drawIndex<64)l.drawNext=starts[l.sharedIndex].uniforms[l.drawIndex];
                else {if(l.sharedIndex>=0&&(shareMode&1)&&l.drawIndex==64){if(!l.rng)l.rng=std::make_unique<std::mt19937_64>();l.rng->seed(l.policySeed);l.rng->discard(64);}
                    l.drawNext=std::generate_canonical<double,std::numeric_limits<double>::digits>(*l.rng);}
                ++l.drawIndex;
            }
            l.ready=true;return;
        }
}
void WaveDiag::prepare(){
    if(awaiting)throw std::logic_error("submit prior scores first");
    auto lock=pool.acquire();
    pool.runIndependent(lanes.size(),[&](std::size_t i){
        prepareLane(i);
    });
    active.clear();for(int i=0;i<int(lanes.size());++i)if(lanes[i].ready){lanes[i].lastRow=active.size();active.push_back(i);}
    awaiting=!active.empty();
}
void WaveDiag::advance(const float* scores,int b,int a){
    if(!awaiting||b!=int(active.size()))throw std::logic_error("bad score count");auto lock=pool.acquire();
    pool.runIndependent(active.size(),[&](std::size_t i){
        auto &l=lanes[active[i]];int n=l.obs.actions.size();if(a>=0&&n>a)throw std::logic_error("short scores");
        thread_local std::vector<float> logits;thread_local std::vector<double> weights;
        auto off=a<0?offset[i]:int(i)*a;logits.assign(scores+off,scores+off+n);for(float x:logits)if(!std::isfinite(x))throw std::logic_error("nonfinite action");
        auto &r=*roots[workRoot[l.task]];auto j=drawPlayout(logits,r.cfg.temperature,weights,*l.rng);
        Action(l.obs.actions[j].bits).execute(*l.state);++l.steps;++l.depth;++l.scored;l.jobActions+=n;
    });awaiting=false;
}
}
