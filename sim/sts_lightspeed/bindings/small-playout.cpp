#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "sim/search/SmallPlayout.h"
#include "sim/search/NeuralSearchBatch.h"
#include <cstring>
namespace py = pybind11;
using namespace sts::search;
namespace {
py::dict features(const SmallFeatures &f) {
    py::array_t<float> state(SMALL_STATE);
    std::memcpy(state.mutable_data(),f.state.data(),SMALL_STATE*sizeof(float));
    const auto n=py::ssize_t(f.actions.size());
    py::array_t<float> actions({n,py::ssize_t(SMALL_ACTION)});
    py::array_t<int> keys({n,py::ssize_t(10)});
    std::fill(keys.mutable_data(),keys.mutable_data()+n*10,-1);
    std::vector<std::uint32_t> bits;
    for(py::ssize_t i=0;i<n;++i){
        const auto &a=f.actions[i]; bits.push_back(a.bits);
        std::memcpy(actions.mutable_data()+i*SMALL_ACTION,a.x.data(),SMALL_ACTION*sizeof(float));
        if(a.keys.size()>10) throw std::logic_error("too many compact action keys");
        std::copy(a.keys.begin(),a.keys.end(),keys.mutable_data()+i*10);
    }
    py::dict d;d["state"]=state;d["actions"]=actions;d["keys"]=keys;d["action_bits"]=bits;return d;
}
py::dict result(const SmallSearchResult &r) {
    auto d=features(r.features);d["action"]=r.action;d["visits"]=r.visits;d["values"]=r.values;
    const auto &s=r.stats;py::dict st;
    st["simulations"]=s.simulations;st["tree_transitions"]=s.treeTransitions;
    st["playout_transitions"]=s.playoutTransitions;st["terminal_wins"]=s.terminalWins;
    st["terminal_losses"]=s.terminalLosses;st["terminal_escapes"]=s.terminalEscapes;
    st["depth_cuts"]=s.depthCuts;st["tree_cuts"]=s.treeCuts;st["chance_replays"]=s.chanceReplays;
    st["nodes"]=s.nodes;st["policy_calls"]=s.policyCalls;st["action_scores"]=s.actionScores;
    st["playout_lengths"]=s.playoutLengths;st["search_seconds"]=s.seconds;d["stats"]=st;return d;
}
}
void bindSmallPlayout(py::module_ &m) {
    m.def("small_playout_key_count",&smallKeyCount);
    m.def("small_playout_features",[](const sts::BattleContext &bc){return features(smallFeatures(bc));});
    m.def("small_playout_value",&smallManualValue);
    py::class_<SmallSearchConfig>(m,"SmallPlayoutConfig")
        .def(py::init<>()).def_readwrite("simulations",&SmallSearchConfig::simulations)
        .def_readwrite("playout_depth",&SmallSearchConfig::playoutDepth)
        .def_readwrite("tree_depth",&SmallSearchConfig::treeDepth)
        .def_readwrite("exploration",&SmallSearchConfig::exploration)
        .def_readwrite("temperature",&SmallSearchConfig::temperature);
    py::class_<SmallPlayoutNet,std::shared_ptr<SmallPlayoutNet>>(m,"SmallPlayoutNet")
        .def(py::init([](py::array_t<float,py::array::c_style|py::array::forcecast> w){
            if(w.ndim()!=1) throw std::invalid_argument("weights must be flat");
            return std::make_shared<SmallPlayoutNet>(std::vector<float>(w.data(),w.data()+w.size()));
        }))
        .def("logits",[](const SmallPlayoutNet &net,const sts::BattleContext &bc){return net.logits(smallFeatures(bc));})
        .def("search",[](const SmallPlayoutNet &net,const sts::BattleContext &bc,const SmallSearchConfig &cfg,std::uint64_t nonce){
            sts::BattleContext root(bc); SmallSearchResult r;
            {py::gil_scoped_release release;r=smallSearch(root,net,cfg,nonce);}
            return result(r);
        });
    // Existing pool owns and joins its worker threads; each call retains all roots/weights.
    m.def("small_playout_search_batch",[](NeuralSearchBatch &pool,const SmallPlayoutNet &net,
            const std::vector<sts::BattleContext*> &roots,const std::vector<SmallSearchConfig> &configs,
            const std::vector<std::uint64_t> &nonces){
        if(roots.size()!=configs.size() || roots.size()!=nonces.size())
            throw std::invalid_argument("compact batch lengths differ");
        std::vector<sts::BattleContext> copies;copies.reserve(roots.size());
        for(auto root:roots){if(!root)throw std::invalid_argument("null compact root");copies.push_back(*root);}
        std::vector<SmallSearchResult> results(roots.size());
        auto lock=pool.acquire();
        {py::gil_scoped_release release;pool.runIndependent(roots.size(),[&](std::size_t i){
            results[i]=smallSearch(copies[i],net,configs[i],nonces[i]);
        });}
        py::list out;for(const auto &r:results)out.append(result(r));return out;
    });
}
