// WaveDiag's persistent native lane scheduler, adapted only at the model boundary.
// Actual game rules remain in the explicitly pinned simulator build.
// PCG uniforms and paired worlds are supplied by the current training contract.
#pragma once
#include "sim/search/NeuralSearchBatch.h"
#include "sim/search/CombatChoices.h"
#include "sim/search/Nosl.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <atomic>
#include <cmath>
#include <memory>

namespace sts::search {
namespace py = pybind11;
template<class T> using AnchorArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

class AnchorWave {
    struct Root {
        BattleContext state;
        int prefix;
        bool draw;
        std::vector<int> actions;
        std::vector<uint32_t> bits;
        std::vector<int64_t> worlds;
        std::vector<double> uniforms;
        Root(const BattleContext& bc, int selected):state(bc),prefix(selected),
            draw(bc.inputState==InputState::CARD_SELECT && combatChoices(bc).zone==2) {
#ifdef STS_ENABLE_RULE_TRACE
            state.atomicStepObserver={};
#endif
            for(const auto& action:autoregressiveActions(state,prefix)) {
                const int id=jointActionId(action,state)-kJointEnd;
                if(id<0)throw std::invalid_argument("native fallback action has no current policy slot");
                actions.push_back(id);bits.push_back(action.bits);
            }
            // Match np.flatnonzero(action_masks), not native enumeration order.
            std::vector<size_t> order(actions.size());
            for(size_t i=0;i<order.size();++i)order[i]=i;
            std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return actions[a]<actions[b];});
            auto ids=actions;auto atoms=bits;
            for(size_t i=0;i<order.size();++i){actions[i]=ids[order[i]];bits[i]=atoms[order[i]];}
        }
    };
    struct Job {int root,action,world;};
    struct Lane {
        std::unique_ptr<BattleContext> state;
        int job=-1,prefix=0,depth=0,cursor=0,kind=0;
        std::vector<Action> actions;
        uint64_t forced=0,policy=0,steps=0,anchor=0,cutoff=0,terminal=0;
    };
    std::vector<Root> roots_;
    std::vector<Job> jobs_;
    std::vector<Lane> lanes_;
    std::vector<double> values_;
    std::vector<int> active_;
    std::atomic<size_t> next_{0};
    NeuralSearchBatch pool_;
    int max_depth_,max_active_=0;
    bool awaiting_=false,closed_=false;

    void step(Lane& lane,const Action& action) {
        executeAutoregressiveAction(action,*lane.state,lane.prefix);
        ++lane.depth;++lane.steps;
    }
    void prepareLane(size_t index) {
        auto& lane=lanes_[index];
        for(;;) {
            if(lane.job<0) {
                const auto task=next_.fetch_add(1);
                if(task>=jobs_.size())return;
                lane.job=int(task);const auto job=jobs_[task];const auto& root=roots_[job.root];
                if(!lane.state)lane.state=std::make_unique<BattleContext>(root.state);
                else *lane.state=root.state; // stable Python view address for the entire lane lifetime
                lane.prefix=root.prefix;lane.depth=0;lane.cursor=0;
                noslRedeterminize(*lane.state,uint64_t(root.worlds[job.world]));
                uint32_t bits=root.bits[job.action];
                if((bits>>29)==2 && root.draw)
                    bits=Action::remapDrawSelection(*lane.state,root.state,bits);
                step(lane,Action(bits));
            }
            auto& state=*lane.state;
            if(state.outcome==Outcome::PLAYER_LOSS) {
                values_[lane.job]=0.;++lane.terminal;lane.job=-1;continue;
            }
            if(state.outcome==Outcome::PLAYER_VICTORY || state.outcome==Outcome::PLAYER_ESCAPE) {
                lane.kind=1;return; // pre-reward anchor; never exit_battle
            }
            if(state.outcome!=Outcome::UNDECIDED)
                throw std::logic_error("unclassified native fallback outcome");
            if(lane.depth>=max_depth_) {lane.kind=2;return;}
            lane.actions=autoregressiveActions(state,lane.prefix);
            if(lane.actions.empty())throw std::logic_error("empty live native fallback actions");
            if(lane.actions.size()==1) {step(lane,lane.actions[0]);++lane.forced;continue;}
            lane.kind=0;return;
        }
    }
public:
    AnchorWave(std::vector<const BattleContext*> states,std::vector<int> prefixes,
            const std::vector<AnchorArray<int64_t>>& worlds,
            const std::vector<AnchorArray<double>>& uniforms,int capacity,int threads,int depth):
            pool_(threads),max_depth_(depth) {
        if(states.empty()||states.size()!=prefixes.size()||states.size()!=worlds.size()
                ||states.size()!=uniforms.size()||capacity<1||depth<1)
            throw std::invalid_argument("invalid native anchor wave");
        for(size_t r=0;r<states.size();++r) {
            if(!states[r]||states[r]->outcome!=Outcome::UNDECIDED||worlds[r].ndim()!=1
                    ||uniforms[r].ndim()!=2||uniforms[r].shape(0)!=worlds[r].size()
                    ||uniforms[r].shape(1)!=depth||!worlds[r].size())
                throw std::invalid_argument("invalid paired native root");
            roots_.emplace_back(*states[r],prefixes[r]);auto& root=roots_.back();
            root.worlds.assign(worlds[r].data(),worlds[r].data()+worlds[r].size());
            root.uniforms.assign(uniforms[r].data(),uniforms[r].data()+uniforms[r].size());
            for(double u:root.uniforms)if(!(u>=0. && u<1.))throw std::invalid_argument("invalid PCG draw");
            for(int w=0;w<int(root.worlds.size());++w)
                for(int a=0;a<int(root.actions.size());++a)jobs_.push_back({int(r),a,w});
        }
        if(jobs_.empty())throw std::invalid_argument("no paired jobs");
        lanes_.resize(std::min(size_t(capacity),jobs_.size()));
        values_.assign(jobs_.size(),std::numeric_limits<double>::quiet_NaN());
    }
    py::array_t<int64_t> ready() {
        if(closed_||awaiting_)throw std::logic_error("native wave has unconsumed scores");
        auto lock=pool_.acquire();
        {py::gil_scoped_release release;
         pool_.runIndependent(lanes_.size(),[&](size_t i){prepareLane(i);});}
        active_.clear();
        for(size_t i=0;i<lanes_.size();++i)if(lanes_[i].job>=0)active_.push_back(int(i));
        max_active_=std::max(max_active_,int(active_.size()));awaiting_=!active_.empty();
        AnchorArray<int64_t> output({ssize_t(active_.size()),ssize_t(6)});auto out=output.mutable_unchecked<2>();
        for(size_t j=0;j<active_.size();++j) {
            auto& lane=lanes_[active_[j]];
            out(j,0)=active_[j];out(j,1)=jobs_[lane.job].root;out(j,2)=lane.job;
            out(j,3)=lane.prefix;out(j,4)=lane.depth;out(j,5)=lane.kind;
        }
        return output;
    }
    BattleContext& state(int index) {
        if(index<0||index>=int(lanes_.size())||!lanes_[index].state)
            throw std::invalid_argument("unknown native lane");
        return *lanes_[index].state;
    }
    void advance(AnchorArray<double> logits,AnchorArray<double> values) {
        if(closed_||!awaiting_||logits.ndim()!=2||logits.shape(0)!=ssize_t(active_.size())
                ||values.ndim()!=1||values.size()!=ssize_t(active_.size()))
            throw std::invalid_argument("native frontier score shape changed");
        const auto width=logits.shape(1);const auto scores=logits.data();const auto leaf=values.data();
        for(ssize_t i=0;i<values.size();++i)if(!std::isfinite(leaf[i]))
            throw std::invalid_argument("nonfinite native leaf value");
        auto lock=pool_.acquire();
        {py::gil_scoped_release release;
         pool_.runIndependent(active_.size(),[&](size_t row) {
            auto& lane=lanes_[active_[row]];
            if(lane.kind) {
                values_[lane.job]=leaf[row];
                if(lane.kind==1)++lane.anchor;else ++lane.cutoff;
                lane.job=-1;return;
            }
            // np.choice consumes only policy decisions, never forced steps.
            const auto& job=jobs_[lane.job];const auto& root=roots_[job.root];
            double maximum=-std::numeric_limits<double>::infinity();
            std::vector<std::pair<int,Action>> actions;
            for(const auto& action:lane.actions) {
                int id=jointActionId(action,*lane.state)-kJointEnd;
                if(id<0||id>=width||!std::isfinite(scores[row*width+id]))
                    throw std::invalid_argument("native policy action/logit mismatch");
                actions.emplace_back(id,action);maximum=std::max(maximum,scores[row*width+id]);
            }
            std::sort(actions.begin(),actions.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            std::vector<double> probabilities;double total=0.;
            for(const auto& action:actions) {
                double p=std::exp(scores[row*width+action.first]-maximum);
                probabilities.push_back(p);total+=p;
            }
            const double u=root.uniforms[job.world*max_depth_+lane.cursor++];
            double cumulative=0.;size_t selected=actions.size()-1;
            for(size_t i=0;i<actions.size();++i) {
                cumulative+=probabilities[i]/total;
                if(u<cumulative){selected=i;break;}
            }
            step(lane,actions[selected].second);++lane.policy;
         });}
        awaiting_=false;
    }
    py::list results() const {
        if(awaiting_||std::any_of(values_.begin(),values_.end(),[](double v){return !std::isfinite(v);}))
            throw std::logic_error("partial native teacher cannot be consumed");
        py::list output;size_t at=0;
        for(const auto& root:roots_) {
            AnchorArray<int64_t> actions(root.actions.size()),visits(root.actions.size());
            AnchorArray<double> means(root.actions.size());
            std::fill(means.mutable_data(),means.mutable_data()+means.size(),0.);
            for(size_t w=0;w<root.worlds.size();++w)
                for(size_t a=0;a<root.actions.size();++a)means.mutable_data()[a]+=values_[at++];
            for(size_t a=0;a<root.actions.size();++a) {
                actions.mutable_data()[a]=root.actions[a];visits.mutable_data()[a]=root.worlds.size();
                means.mutable_data()[a]/=root.worlds.size();
            }
            py::dict row;row["actions"]=actions;row["visits"]=visits;row["values"]=means;output.append(row);
        }
        return output;
    }
    py::dict statistics() const {
        uint64_t forced=0,policy=0,steps=0,anchor=0,cutoff=0,terminal=0;
        for(const auto& l:lanes_){forced+=l.forced;policy+=l.policy;steps+=l.steps;
            anchor+=l.anchor;cutoff+=l.cutoff;terminal+=l.terminal;}
        py::dict result;result["chains"]=jobs_.size();result["atomic_steps"]=steps;
        result["forced_rows"]=forced;result["policy_rows"]=policy;result["anchor_rows"]=anchor;
        result["cutoff_rows"]=cutoff;result["terminal_rows"]=terminal;result["max_active"]=max_active_;
        return result;
    }
};
inline void bindAnchorWave(py::module_& module) {
    py::class_<AnchorWave>(module,"AnchorWave")
        .def(py::init<std::vector<const BattleContext*>,std::vector<int>,
            const std::vector<AnchorArray<int64_t>>&,const std::vector<AnchorArray<double>>&,
            int,int,int>())
        .def("ready",&AnchorWave::ready).def("advance",&AnchorWave::advance)
        .def("state",&AnchorWave::state,py::return_value_policy::reference_internal)
        .def("results",&AnchorWave::results).def("statistics",&AnchorWave::statistics);
}
}
