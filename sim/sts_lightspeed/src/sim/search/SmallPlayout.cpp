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
void featuresForActions(const BattleContext &bc, const std::vector<Action> &actions, SmallFeatures &f) {
    // Every state field is assigned below; retain only the actions' capacity.
    f.actions.clear();
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
