#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "sim/search/SmallPlayout.h"
#include "sim/search/PlayoutEntities.h"
#include "sim/search/PublicPlayoutAdapter.h"
#include "game/GameContext.h"
#include "sim/search/NeuralSearchBatch.h"
#include <cstring>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
namespace py = pybind11;
using namespace sts::search;
namespace {
py::dict entityRows(const std::vector<PlayoutEntity> &rows) {
    const auto n=py::ssize_t(rows.size());
    py::array_t<int> ids(n), zones(n), monsters(n);
    py::array_t<float> numbers({n,py::ssize_t(PLAYOUT_ENTITY_FLOATS)});
    for(py::ssize_t i=0;i<n;++i){
        ids.mutable_data()[i]=rows[i].id;zones.mutable_data()[i]=rows[i].zone;
        monsters.mutable_data()[i]=rows[i].monster;
        std::copy(rows[i].numbers.begin(),rows[i].numbers.end(),numbers.mutable_data()+i*PLAYOUT_ENTITY_FLOATS);
    }
    py::dict d;d["ids"]=ids;d["zones"]=zones;d["monsters"]=monsters;d["numbers"]=numbers;return d;
}
py::dict features(const SmallFeatures &f) {
    py::array_t<float> state(SMALL_STATE);
    std::memcpy(state.mutable_data(),f.state.data(),SMALL_STATE*sizeof(float));
    const auto n=py::ssize_t(f.actions.size());
    py::array_t<float> actions({n,py::ssize_t(SMALL_ACTION)});
    py::array_t<int> keys({n,py::ssize_t(SMALL_KEYS)});
    std::fill(keys.mutable_data(),keys.mutable_data()+n*SMALL_KEYS,-1);
    std::vector<std::uint32_t> bits;
    for(py::ssize_t i=0;i<n;++i){
        const auto &a=f.actions[i]; bits.push_back(a.bits);
        std::memcpy(actions.mutable_data()+i*SMALL_ACTION,a.x.data(),SMALL_ACTION*sizeof(float));
        std::copy_n(a.keys.begin(),a.keyCount,keys.mutable_data()+i*SMALL_KEYS);
    }
    py::dict d;d["state"]=state;d["actions"]=actions;d["keys"]=keys;d["action_bits"]=bits;return d;
}
py::dict result(const SmallSearchResult &r) {
    auto d=features(r.features);d["action"]=r.action;d["visits"]=r.visits;d["values"]=r.values;
    d["terminal_visits"]=r.terminalVisits;
    d["value_statistic"]="terminal-return-lower-bound";
    const auto &s=r.stats;py::dict st;
    st["simulations"]=s.simulations;st["tree_transitions"]=s.treeTransitions;
    st["playout_transitions"]=s.playoutTransitions;st["terminal_wins"]=s.terminalWins;
    st["terminal_losses"]=s.terminalLosses;st["terminal_escapes"]=s.terminalEscapes;
    st["depth_cuts"]=s.depthCuts;st["tree_cuts"]=s.treeCuts;st["chance_replays"]=s.chanceReplays;
    st["nodes"]=s.nodes;st["policy_calls"]=s.policyCalls;st["action_scores"]=s.actionScores;
    st["forced_policy_skips"]=s.forcedPolicySkips;
    if(s.publicStateRefreshes)st["public_state_refreshes"]=s.publicStateRefreshes;
    st["prior_calls"]=s.priorCalls;st["draw_edges"]=s.drawEdges;
    st["guard_transitions"]=s.guardTransitions;
    st["turn_card_limit_decisions"]=s.turnCardLimitDecisions;
    st["turn_card_limit_simulated"]=s.turnCardLimitSimulated;
    st["playout_lengths"]=s.playoutLengths;st["search_seconds"]=s.seconds;d["stats"]=st;return d;
}
}
void bindSmallPlayout(py::module_ &m) {
    m.attr("small_playout_search_contract")="terminal-hp-boss-win-turn128-v1";
    m.attr("small_paired_search_contract")="paired-worldlines-eps005-terminal-hp-boss-win-turn128-v1";
    m.attr("small_paired_world_max_simulations")=4096;
    m.attr("small_playout_max_cards_per_turn")=SMALL_MAX_CARDS_PER_TURN;
    m.def("small_playout_turn_card_limit_reached",&smallTurnCardLimitReached);
    m.def("small_playout_key_count",&smallKeyCount);
    m.def("small_playout_features",[](const sts::BattleContext &bc){return features(smallFeatures(bc));});
    m.attr("small_playout_public_contract")="sts1-playout-public-entities-v1";
    m.def("small_playout_public_layout",[]{
        py::dict d;d["stride"]=PLAYOUT_ENTITY_STRIDE;d["numbers"]=PLAYOUT_ENTITY_FLOATS;
        d["kinds"]=py::make_tuple("battle","player","power","relic","card","monster","monster_power",
            "move","potion","selection","counter","order");
        d["zones"]=py::make_tuple("battle","player","hand","hidden_draw","known_draw","discard","exhaust",
            "stasis","generated","pending","master_deck","owned_relic","active_relic","potion","monster");
        return d;
    });
    m.def("small_playout_public_features",[](const sts::GameContext &gc,const sts::BattleContext &bc){
        const auto state=smallFeatures(bc);
        PlayoutEntityObservation observation;
        playoutEntities(bc,gc.deck.size(),state.actions,observation);
        auto d=features(state);
        d["inventory"]=entityRows(playoutInventory(gc));
        d["entities"]=entityRows(observation.entities);
        const auto n=py::ssize_t(observation.sources.size());
        py::array_t<int> sources({n,py::ssize_t(SMALL_KEYS)}),targets(n);
        for(py::ssize_t i=0;i<n;++i){
            std::copy_n(observation.sources[i].begin(),SMALL_KEYS,sources.mutable_data()+i*SMALL_KEYS);
            targets.mutable_data()[i]=observation.targets[i];
        }
        d["source_rows"]=sources;d["target_slots"]=targets;return d;
    },py::arg("game"),py::arg("battle"));
    m.def("small_playout_value",&smallTerminalValue,py::arg("battle"),py::arg("boss")=false);
    m.def("small_playout_redeterminize",&smallRedeterminize);
    m.def("small_playout_known_top",[](const sts::BattleContext &bc){return bc.cards.knownDrawTop;});
    m.def("small_paired_worldlines",[](const sts::BattleContext &root, const SmallPlayoutNet &net,
            const SmallSearchConfig &cfg, const std::vector<std::uint32_t> &actions, int budget, std::uint64_t nonce){
        PairedWorldlineResult out;
        { py::gil_scoped_release release; out=smallPairedWorldlines(root,net,cfg,actions,budget,nonce); }
        py::dict d; d["values"]=out.values; d["completed"]=out.completed;
        d["root_transitions"]=budget; d["playout_transitions"]=out.stats.playoutTransitions-budget; d["policy_calls"]=out.stats.policyCalls;
        return d;
    });
    m.def("small_playout_root_pick",&smallRootPick);
    py::class_<PublicPlayoutAdapter,std::shared_ptr<PublicPlayoutAdapter>>(m,"PublicPlayoutAdapter")
        .def(py::init([](py::array_t<float,py::array::c_style|py::array::forcecast> w){
            if(w.ndim()!=1)throw std::invalid_argument("public context weights must be flat");
            return std::make_shared<PublicPlayoutAdapter>(std::vector<float>(w.data(),w.data()+w.size()));
        }));
    py::class_<SmallSearchConfig>(m,"SmallPlayoutConfig")
        .def("condition",[](SmallSearchConfig &cfg,const std::shared_ptr<PublicPlayoutAdapter> &adapter,
                           const sts::GameContext &gc,const sts::BattleContext &bc,bool refresh){
            if(!adapter)throw std::invalid_argument("missing public adapter");
            cfg.publicRoot=PublicPlayoutRoot::prepare(adapter,gc,bc,refresh ? ContextRefresh::State : ContextRefresh::Root);
        })
        .def("condition_turns",[](SmallSearchConfig &cfg,const std::shared_ptr<PublicPlayoutAdapter> &adapter,
                           const sts::GameContext &gc,const sts::BattleContext &bc){
            if(!adapter)throw std::invalid_argument("missing public adapter");
            cfg.publicRoot=PublicPlayoutRoot::prepare(adapter,gc,bc,ContextRefresh::Turn);
        })
        .def_property_readonly("public_context",[](const SmallSearchConfig &cfg){
            if(!cfg.publicRoot)throw std::logic_error("unconditioned root");
            py::array_t<float> out(PUBLIC_SKETCH_SIZE);
            std::copy_n(cfg.publicRoot->sketch.begin(),PUBLIC_SKETCH_SIZE,out.mutable_data());return out;
        })
        .def(py::init<>()).def_readwrite("paired_worldlines",&SmallSearchConfig::pairedWorldlines).def_readwrite("simulations",&SmallSearchConfig::simulations)
        .def_readwrite("playout_depth",&SmallSearchConfig::playoutDepth)
        .def_readwrite("tree_depth",&SmallSearchConfig::treeDepth)
        .def_readwrite("exploration",&SmallSearchConfig::exploration)
        .def_readwrite("temperature",&SmallSearchConfig::temperature)
        .def_readwrite("use_prior",&SmallSearchConfig::usePrior)
        .def_readwrite("prior_mix",&SmallSearchConfig::priorMix)
        .def_readwrite("prior_temperature",&SmallSearchConfig::priorTemperature)
        .def_readwrite("boss",&SmallSearchConfig::boss);
    py::class_<SmallPlayoutNet,std::shared_ptr<SmallPlayoutNet>>(m,"SmallPlayoutNet")
        .def(py::init([](py::array_t<float,py::array::c_style|py::array::forcecast> w){
            if(w.ndim()!=1) throw std::invalid_argument("weights must be flat");
            return std::make_shared<SmallPlayoutNet>(std::vector<float>(w.data(),w.data()+w.size()));
        }))
        .def("conditioned_logits",[](const SmallPlayoutNet &net,const sts::BattleContext &bc,const SmallSearchConfig &cfg){
            if(!cfg.publicRoot)throw std::logic_error("unconditioned root");
            std::vector<float> out;
            if(cfg.publicRoot->refresh!=ContextRefresh::Root){
                const auto shift=cfg.publicRoot->liveShift(bc);
                net.logits(smallFeatures(bc),out,shift.data());
            }else net.logits(smallFeatures(bc),out,cfg.publicRoot->shift.data());
            return out;
        })
        .def("logits",[](const SmallPlayoutNet &net,const sts::BattleContext &bc){return net.logits(smallFeatures(bc));})
        .def("search",[](const SmallPlayoutNet &net,const sts::BattleContext &bc,const SmallSearchConfig &cfg,std::uint64_t nonce,const std::shared_ptr<SmallPlayoutNet> &prior){
            sts::BattleContext root(bc); SmallSearchResult r;
            {py::gil_scoped_release release;r=smallSearch(root,net,cfg,nonce,prior.get());}
            return result(r);
        },py::arg("root"),py::arg("config"),py::arg("nonce"),py::arg("prior")=nullptr);
    // Existing pool owns and joins its worker threads; each call retains all roots/weights.
    m.def("small_playout_search_batch",[](NeuralSearchBatch &pool,const SmallPlayoutNet &net,
            const std::vector<sts::BattleContext*> &roots,const std::vector<SmallSearchConfig> &configs,
            const std::vector<std::uint64_t> &nonces,const std::shared_ptr<SmallPlayoutNet> &prior,
            const py::object &onReady,const py::object &onStart){
        if(roots.size()!=configs.size() || roots.size()!=nonces.size())
            throw std::invalid_argument("compact batch lengths differ");
        const bool ready=!onReady.is_none(), start=!onStart.is_none();
        if((ready && !PyCallable_Check(onReady.ptr())) || (start && !PyCallable_Check(onStart.ptr())))
            throw std::invalid_argument("compact callbacks must be callable");
        std::vector<sts::BattleContext> copies;copies.reserve(roots.size());
        for(auto root:roots){if(!root)throw std::invalid_argument("null compact root");copies.push_back(*root);}
        std::vector<SmallSearchResult> results(roots.size());
        auto lock=pool.acquire();
        if(!ready && !start) {
            py::gil_scoped_release release;
            pool.runIndependent(roots.size(),[&](std::size_t i){
                results[i]=smallSearch(copies[i],net,configs[i],nonces[i],prior.get());
            });
        } else {
            // Native workers publish completion without waiting for the GIL.
            // One consumer owns Python callbacks, avoiding GIL contention that
            // otherwise stalls all search workers at once. Roots and weights
            // remain retained until both the pool and consumer have joined.
            std::mutex mutex;
            std::condition_variable available;
            std::deque<std::size_t> completed;
            bool finished=false;
            std::exception_ptr searchFailure, callbackFailure;
            {
                py::gil_scoped_release release;
                std::thread consumer([&]{
                    py::gil_scoped_acquire acquire;
                    try {
                        if(start) onStart();
                        while(true) {
                            std::size_t i=0;
                            {
                                py::gil_scoped_release waitRelease;
                                std::unique_lock<std::mutex> guard(mutex);
                                available.wait(guard,[&]{return finished || !completed.empty();});
                                if(completed.empty()) break;
                                i=completed.front();completed.pop_front();
                            }
                            if(ready) onReady(i,results[i].action);
                        }
                    } catch(...) { callbackFailure=std::current_exception(); }
                });
                try {
                    pool.runIndependent(roots.size(),[&](std::size_t i){
                        results[i]=smallSearch(copies[i],net,configs[i],nonces[i],prior.get());
                        {std::lock_guard<std::mutex> guard(mutex);completed.push_back(i);}
                        available.notify_one();
                    });
                } catch(...) { searchFailure=std::current_exception(); }
                {std::lock_guard<std::mutex> guard(mutex);finished=true;}
                available.notify_one();
                consumer.join();
            }
            if(callbackFailure) std::rethrow_exception(callbackFailure);
            if(searchFailure) std::rethrow_exception(searchFailure);
        }
        py::list out;for(const auto &r:results)out.append(result(r));return out;
    },py::arg("pool"),py::arg("net"),py::arg("roots"),py::arg("configs"),py::arg("nonces"),py::arg("prior")=nullptr,
       py::arg("on_ready")=py::none(),py::arg("on_start")=py::none());
    m.def("small_playout_world_batch",[](NeuralSearchBatch &pool,const SmallPlayoutNet &net,
            const std::vector<sts::BattleContext*> &roots,const std::vector<SmallSearchConfig> &configs,
            const std::vector<std::uint64_t> &nonces,const std::shared_ptr<SmallPlayoutNet> &prior){
        if(roots.size()!=configs.size()||roots.size()!=nonces.size()||roots.size()>480||prior)
            throw std::invalid_argument("unsupported world batch inputs");
        std::vector<sts::BattleContext> copies;copies.reserve(roots.size());
        for(auto root:roots){if(!root)throw std::invalid_argument("null world root");copies.push_back(*root);}
        std::vector<SmallSearchResult> results;auto lock=pool.acquire();
        {py::gil_scoped_release release;results=smallPairedWorldBatch(pool,copies,net,configs,nonces);}
        py::list out;for(const auto &r:results)out.append(result(r));return out;
    },py::arg("pool"),py::arg("net"),py::arg("roots"),py::arg("configs"),py::arg("nonces"),py::arg("prior")=nullptr);
    m.attr("small_playout_parallel_steps")=true;
    m.def("small_playout_collect_batch",[](NeuralSearchBatch &pool,const SmallPlayoutNet &net,
            const std::vector<sts::BattleContext*> &roots,const SmallSearchConfig &cfg,
            const std::vector<std::uint64_t> &nonces){
        if(roots.size()!=nonces.size())throw std::invalid_argument("playout batch lengths differ");
        std::vector<sts::BattleContext> copies;for(auto root:roots)copies.push_back(*root);
        std::vector<SmallPlayoutTrace> traces(roots.size());auto lock=pool.acquire();
        {py::gil_scoped_release release;pool.runIndependent(roots.size(),[&](std::size_t i){
            traces[i]=smallCollectPlayout(copies[i],net,cfg,nonces[i]);
        });}
        py::list out;
        for(const auto &t:traces){
            py::dict d;py::list steps;
            for(std::size_t i=0;i<t.features.size();++i){auto f=features(t.features[i]);f["choice"]=t.choices[i];steps.append(f);}
            d["steps"]=steps;
            d["value"]=std::isfinite(t.value) ? py::cast(t.value) : py::none();
            d["outcome"]=t.outcome;d["censored"]=t.stats.depthCuts>0;
            d["transitions"]=t.stats.playoutTransitions;out.append(d);
        }
        return out;
    });
}
