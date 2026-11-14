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
#include "sim/search/WaveDiag.h"
#include "compact-observation.h"
namespace {
template<class T> py::array_t<T> wa(std::initializer_list<int> shape){std::vector<py::ssize_t> v(shape.begin(),shape.end());return py::array_t<T>(v);}
py::object wavePacket(WaveDiag &batch) {
    {py::gil_scoped_release release;batch.prepare();}
    const int b=batch.active.size();if(!b)return py::none();
    int a=0,n=0,t=0,q=0;std::vector<int> offset(b);
    for(int i=0;i<b;++i){auto &l=batch.lanes[batch.active[i]];
        offset[i]=n;n+=l.obs.actions.size();a=std::max(a,int(l.obs.actions.size()));
        t=std::max(t,int(l.live.size()));q=std::max(q,int(l.queries.size()));}
    auto state=wa<float>({b,32}),act=wa<float>({n,24}),feat=wa<float>({b,t,128}),ctx=wa<float>({b,1552});
    auto keys=wa<int>({n,10}),slots=wa<int>({n}),sym=wa<int>({b,t+2}),ids=wa<int>({b,t+2}),query=wa<int>({b,q}),src=wa<int>({n,10}),target=wa<int>({n});
    auto mask=wa<bool>({b,a}),present=wa<bool>({b,t+2});
    float *st=state.mutable_data(),*ac=act.mutable_data(),*ft=feat.mutable_data(),*cx=ctx.mutable_data();
    int *ky=keys.mutable_data(),*sl=slots.mutable_data(),*sy=sym.mutable_data(),*id=ids.mutable_data(),*qu=query.mutable_data(),*sc=src.mutable_data(),*tg=target.mutable_data();
    bool *mk=mask.mutable_data(),*pr=present.mutable_data();
    {py::gil_scoped_release release;auto lock=batch.pool.acquire();batch.pool.runIndependent(b,[&](std::size_t i){
        auto &l=batch.lanes[batch.active[i]];int m=l.obs.actions.size(),o=offset[i];
        std::copy_n(l.obs.state.data(),32,st+i*32);if(l.ctx.empty())std::fill_n(cx+i*1552,1552,0.f);else std::copy_n(l.ctx.data(),1552,cx+i*1552);
        std::fill_n(mk+i*a,m,true);std::fill_n(mk+i*a+m,a-m,false);
        std::fill_n(qu+i*q,q,0);std::copy(l.queries.begin(),l.queries.end(),qu+i*q);
        auto p=i*(t+2);std::fill_n(id+p,t+2,0);std::fill_n(sy+p,t+2,0);std::fill_n(pr+p,t+2,false);
        for(int j=0;j<int(l.live.size());++j){int idx=l.live[j];id[p+j]=std::min(33,idx);sy[p+j]=l.symbols[idx];pr[p+j]=true;
            std::copy_n(l.features[idx].data(),128,ft+(i*t+j)*128);}
        std::fill_n(ft+(i*t+l.live.size())*128,(t-l.live.size())*128,0.f);
        id[p+t]=3;id[p+t+1]=1;pr[p+t]=l.features[3][127]>0||l.features[4][127]>0||l.features[5][127]>0;pr[p+t+1]=true;
        for(int j=0;j<m;++j){auto &af=l.obs.actions[j];sl[o+j]=i*a+j;
            std::copy_n(af.x.data(),24,ac+(o+j)*24);std::fill_n(ky+(o+j)*10,10,-1);std::copy_n(af.keys.data(),af.keyCount,ky+(o+j)*10);
            std::copy_n(l.src[j].data(),10,sc+(o+j)*10);tg[o+j]=l.target[j];}
    });}
    return py::make_tuple(state,act,keys,slots,mask,feat,sym,ids,present,query,src,target,ctx);
}
void bindWave(py::module_ &m){
    py::class_<WaveRoot,std::shared_ptr<WaveRoot>>(m,"WaveRoot")
        .def(py::init<const sts::GameContext&,const sts::BattleContext&,SmallSearchConfig,std::uint64_t>())
        .def(py::init<const sts::GameContext&,const sts::BattleContext&,SmallSearchConfig,std::uint64_t,bool>())
        .def("configure_paired",&WaveRoot::configurePaired)
        .def_readwrite("inventory_key",&WaveRoot::keyForInventory).def("inventory_rows",[](WaveRoot &r){return entityRows(r.inventory);});
    py::class_<WaveDiag>(m,"WaveDiag")
        .def(py::init<std::vector<std::shared_ptr<WaveRoot>>,int,int>())
        .def("pack_raw",[](WaveDiag &b,py::dict obs,py::array_t<std::int64_t,py::array::c_style> rows){
            // Current token consumers retain the original native pool/pack
            // boundary. Python must not rebuild each lane's query layout.
            if(!b.awaiting||rows.ndim()!=1||!rows.size())throw std::logic_error("invalid paired raw rows");
            constexpr auto flags=py::array::c_style|py::array::forcecast;
            py::array_t<float,flags> legal(obs["combat_legal"]);
            py::array_t<std::int64_t,flags> hand(obs["hand_ids"]),monster(obs["monster_id"]),potion(obs["potions"]),select(obs["select_native_indices"]);
            const int n=rows.size(),total=legal.shape(0);
            if(legal.ndim()!=2||legal.shape(1)!=198||hand.size()!=total*10||monster.size()!=total*5||potion.size()!=total*5||select.size()!=total*96)
                throw std::logic_error("invalid paired token schema");
            struct Layout {std::array<bool,121> present{},query{};int tokens=0,queries=0,actions=0,choice=1;};
            std::vector<Layout> layout(n);std::vector<std::array<int,5>> offset(n);
            for(int i=0;i<n;++i)if(rows.data()[i]<0||rows.data()[i]>=total)throw std::logic_error("paired row outside batch");
            int nt=0,nq=0,na=0,nb1=0,nb2=0,choice=1;
            {py::gil_scoped_release release;auto lock=b.pool.acquire();
                b.pool.runIndependent(n,[&](std::size_t i){
                    auto &l=layout[i];const int r=rows.data()[i];const float *mask=legal.data()+r*198;
                    l.present[0]=l.query[0]=true;
                    for(int j=0;j<10;++j)l.present[1+j]=hand.data()[r*10+j]!=0;
                    for(int j=0;j<5;++j){l.present[11+j]=monster.data()[r*5+j]!=0;l.present[16+j]=potion.data()[r*5+j]!=0;}
                    for(int j=21;j<25;++j)l.present[j]=true;
                    for(int j=0;j<96;++j){l.present[25+j]=select.data()[r*96+j]>0;if(l.present[25+j]||mask[91+j])l.choice=j+1;}
                    for(int a=0;a<198;++a)if(mask[a]){
                        ++l.actions;
                        if(a>=1&&a<91){int source=(a-1)/6,target=(a-1)%6;l.query[source<10?1+source:16+source-10]=true;if(target<5)l.query[11+target]=true;}
                        else if(a>=91&&a<187)l.query[25+a-91]=true;
                        else if(a>=187)for(int j=0;j<10;++j)if(l.present[1+j])l.query[1+j]=true;
                    }
                    for(int j=0;j<121;++j){l.tokens+=l.present[j]||l.query[j];l.queries+=l.query[j];}
                });
                for(int i=0;i<n;++i){const auto &l=layout[i];if(!l.actions)throw std::logic_error("empty paired action row");
                    offset[i]={nt,nq,na,nb1,nb2};nt+=l.tokens;nq+=l.queries;na+=l.actions;nb1+=(l.tokens+15)/16;nb2+=(l.queries+15)/16;choice=std::max(choice,l.choice);}
            }
            auto retained=wa<int>({nt}),blocks1=wa<int>({nb1,2}),blocks2=wa<int>({nb2,2}),mapping=wa<int>({n,161});
            auto queries=wa<std::int64_t>({nq}),starts=wa<std::int64_t>({n,4}),lens=wa<std::int64_t>({n,4}),keys=wa<std::int64_t>({na,10});
            auto present=wa<bool>({nt});auto owners=wa<int>({na}),lengths=wa<int>({n}),offsets=wa<int>({n}),counts=wa<int>({6});
            std::fill_n(mapping.mutable_data(),mapping.size(),-1);std::fill_n(keys.mutable_data(),keys.size(),-1);
            const int sizes[6]={nt,nq,nb1,nb2,n,na};std::copy_n(sizes,6,counts.mutable_data());
            {py::gil_scoped_release release;auto lock=b.pool.acquire();constexpr int grain=8;
                b.pool.runIndependent((n+grain-1)/grain,[&](std::size_t group){
                    for(int i=group*grain;i<std::min(n,int(group+1)*grain);++i){
                        const auto &l=layout[i];const auto &o=offset[i];const int r=rows.data()[i];
                        const int begin[4]={o[0],o[1],o[3],o[4]},size[4]={l.tokens,l.queries,(l.tokens+15)/16,(l.queries+15)/16};
                        std::copy_n(begin,4,starts.mutable_data()+i*4);std::copy_n(size,4,lens.mutable_data()+i*4);
                        int t=0,q=0,a=0;
                        for(int j=0;j<121;++j)if(l.present[j]||l.query[j]){
                            retained.mutable_data()[o[0]+t]=i*128+j;present.mutable_data()[o[0]+t]=l.present[j];
                            if(l.query[j]){queries.mutable_data()[o[1]+q]=o[0]+t;mapping.mutable_data()[i*161+j]=q++;}++t;
                        }
                        for(int j=0;j<198;++j)if(legal.data()[r*198+j]){keys.mutable_data()[(o[2]+a)*10]=j;owners.mutable_data()[o[2]+a]=i;++a;}
                        for(int j=0;j<size[2];++j){blocks1.mutable_data()[(o[3]+j)*2]=i;blocks1.mutable_data()[(o[3]+j)*2+1]=j*16;}
                        for(int j=0;j<size[3];++j){blocks2.mutable_data()[(o[4]+j)*2]=i;blocks2.mutable_data()[(o[4]+j)*2+1]=j*16;}
                        lengths.mutable_data()[i]=l.actions;offsets.mutable_data()[i]=o[2];
                    }
                });
            }
            py::dict out;out["retained"]=retained;out["queries"]=queries;out["present"]=present;out["starts"]=starts;out["lens"]=lens;
            out["blocks1"]=blocks1;out["blocks2"]=blocks2;out["mapping"]=mapping;out["keys"]=keys;out["owners"]=owners;
            out["lengths"]=lengths;out["offsets"]=offsets;out["counts"]=counts;out["choice"]=choice;return out;
        })
        .def("compact",[](WaveDiag&b,py::array_t<std::int32_t,py::array::c_style> cardIndex){
            using namespace compact_leaf;
            if(!b.awaiting||cardIndex.ndim()!=1||cardIndex.size()!=kCardIds)throw std::logic_error("invalid compact request");
            const int n=b.active.size();
            auto cat=wa<int>({n,kCatSize});auto sc=wa<float>({n,kFloatSize});
            auto ids=wa<int>({n,2,kLanguageSlots});auto feats=wa<float>({n,2,kLanguageSlots,3});
            auto root=wa<int>({n}),prefix=wa<int>({n}),kind=wa<int>({n});
            auto legal=wa<int>({n,kAutoregressiveSize});auto bits=wa<unsigned>({n,kAutoregressiveSize});
            auto draws=wa<double>({n});auto misc=wa<float>({n,sts::Deck::MAX_SIZE});
            auto upgraded=wa<float>({n,3,kCardIds}),upgradeSums=wa<float>({n,3,kCardIds});
            std::fill_n(upgraded.mutable_data(),upgraded.size(),0.f);std::fill_n(upgradeSums.mutable_data(),upgradeSums.size(),0.f);
            std::fill_n(ids.mutable_data(),ids.size(),0);std::fill_n(feats.mutable_data(),feats.size(),0.f);
            std::fill_n(legal.mutable_data(),legal.size(),-1);std::fill_n(bits.mutable_data(),bits.size(),0);
            std::fill_n(misc.mutable_data(),misc.size(),std::numeric_limits<float>::quiet_NaN());
            const auto lut=cardIndex.data();
            {py::gil_scoped_release release;auto lock=b.pool.acquire();b.pool.runIndependent(n,[&](std::size_t i){
                auto&l=b.lanes[b.active[i]];auto&s=*l.state;auto&r=*b.roots[b.workRoot[l.task]];
                if(!r.pairedAnchor)throw std::logic_error("compact transport requires paired contract");
                const auto packed=pack(s);std::copy(packed.categorical.begin(),packed.categorical.end(),cat.mutable_data()+i*kCatSize);
                std::copy(packed.scalars.begin(),packed.scalars.end(),sc.mutable_data()+i*kFloatSize);
                packLanguagePile(s.cards.drawPile,lut,ids.mutable_data()+i*2*kLanguageSlots,feats.mutable_data()+i*6*kLanguageSlots);
                packLanguagePile(s.cards.discardPile,lut,ids.mutable_data()+(i*2+1)*kLanguageSlots,feats.mutable_data()+(i*2+1)*3*kLanguageSlots);
                root.mutable_data()[i]=b.workRoot[l.task];prefix.mutable_data()[i]=l.prefix;kind.mutable_data()[i]=l.leafKind;draws.mutable_data()[i]=l.drawNext;
                auto acts=l.leafKind==4?autoregressiveActions(s,l.prefix):std::vector<sts::search::Action>();
                if(!l.leafKind)for(const auto&a:l.obs.actions)acts.emplace_back(a.bits);
                for(std::size_t j=0;j<acts.size();++j){legal.mutable_data()[i*kAutoregressiveSize+j]=jointActionId(acts[j],s)-kJointEnd;bits.mutable_data()[i*kAutoregressiveSize+j]=acts[j].bits;}
                auto add=[&](const sts::CardInstance&c){if(c.id==sts::CardId::RITUAL_DAGGER&&c.uniqueId>=0&&c.uniqueId<sts::Deck::MAX_SIZE)misc.mutable_data()[i*sts::Deck::MAX_SIZE+c.uniqueId]=float(c.specialData)/100.f;};
                auto addPile=[&](const auto&pile,int zone){for(const auto&c:pile){add(c);const auto at=(i*3+zone)*kCardIds+int(c.id);upgraded.mutable_data()[at]+=c.getUpgradeCount()>0;upgradeSums.mutable_data()[at]+=c.getUpgradeCount();}};
                addPile(s.cards.drawPile,0);addPile(s.cards.discardPile,1);addPile(s.cards.exhaustPile,2);
                for(int j=0;j<s.cards.cardsInHand;++j)add(s.cards.hand[j]);
            });}
            py::dict out;out["categorical"]=cat;out["scalars"]=sc;out["language_ids"]=ids;out["language_features"]=feats;
            out["root_indices"]=root;out["selection_prefix"]=prefix;out["leaf_kind"]=kind;out["legal_policy_ids"]=legal;
            out["action_bits"]=bits;out["draws"]=draws;out["permanent_misc"]=misc;
            out["pile_upgraded_counts"]=upgraded;out["pile_upgrade_sums"]=upgradeSums;return out;
        })
        .def("choose_paired",[](WaveDiag&b,py::array_t<int,py::array::c_style>picks,py::array_t<double,py::array::c_style>values){
            const auto n=b.active.size();if(!b.awaiting||!b.gpuChoice||picks.ndim()!=1||values.ndim()!=1||picks.size()!=n||values.size()!=n)throw std::logic_error("invalid paired reply");
            for(std::size_t i=0;i<n;++i){const auto&l=b.lanes[b.active[i]];
                if(l.leafKind?!std::isfinite(values.data()[i]):(picks.data()[i]<0||picks.data()[i]>=int(l.obs.actions.size())))throw std::logic_error("invalid paired reply row");}
            b.pendingPicks.assign(picks.data(),picks.data()+n);b.pendingValues.assign(values.data(),values.data()+n);b.awaiting=false;
        })
        .def("paired_results",[](WaveDiag&b){
            if(b.awaiting||b.nextTask.load()<int(b.local.size()))throw std::logic_error("unfinished paired search");
            for(const auto&l:b.lanes)if(l.task>=0)throw std::logic_error("live paired lane");
            py::list out;
            for(std::size_t i=0;i<b.roots.size();++i){auto&r=*b.roots[i];const int n=r.actions.size();
                std::vector<int> visits(n),terminal(n),hist(201);std::vector<double> values(n);std::vector<unsigned>bits;
                std::array<unsigned long long,12>stats{};
                for(std::size_t t=0;t<b.local.size();++t)if(b.workRoot[t]==int(i)){
                    if(!std::isfinite(b.values[t]))throw std::logic_error("unscored paired leaf");
                    const int a=b.chosen[t],depth=b.depths[t],o=b.outcomes[t];++visits[a];terminal[a]+=o!=4;values[a]+=b.values[t];
                    stats[0]+=depth-1;stats[1]+=depth-1-b.forceN[t];stats[2]+=b.forceN[t];stats[3]+=o==1;stats[4]+=o==2;stats[5]+=o==4;stats[7]+=depth;stats[8]+=b.maxActionN[t];stats[9]+=o==3;stats[10]+=b.actionN[t];stats[11]+=b.prefixN[t];++hist[depth];
                }
                for(int j=0;j<n;++j){if(visits[j]!=int(r.worlds.size()))throw std::logic_error("unpaired result");values[j]/=visits[j];bits.push_back(r.actions[j].bits);}
                py::dict d;d["action_bits"]=bits;d["values"]=values;d["visits"]=visits;d["terminal_visits"]=terminal;d["statistics"]=stats;
                d["simulations"]=r.cfg.simulations;d["nonforced_decisions"]=stats[1];d["forced_actions"]=stats[2];d["actual_transitions"]=stats[7]-stats[11];d["atomic_decisions"]=stats[7];d["selection_prefix_decisions"]=stats[11];d["depth_cuts"]=stats[5];d["playout_lengths"]=hist;d["value_statistic"]="paired-pcg64-pre-reward-anchor-v1";out.append(d);
            }return out;
        })
        .def("packet",&wavePacket)
        .def("advance",[](WaveDiag &b,py::array_t<float,py::array::c_style> logits){
            if(logits.ndim()!=2)throw std::invalid_argument("expected logits matrix");auto v=logits.data();auto nr=logits.shape(0),nc=logits.shape(1);
            py::gil_scoped_release noGil;b.advance(v,nr,nc);
        })
        .def("raw",[](WaveDiag &b,int row){
            auto &l=b.lanes[b.active.at(row)];auto &r=*b.roots[b.workRoot[l.task]];
            PlayoutEntityObservation e;playoutEntities(*l.state,r.deck,l.obs.actions,e);
            auto d=(b.shortAtoms&&l.state->inputState==sts::InputState::PLAYER_NORMAL)?features(smallFeatures(*l.state)):features(l.obs);d["entities"]=entityRows(e.entities);d["inventory"]=entityRows(r.inventory);
            auto sources=wa<int>({int(e.sources.size()),10}),targets=wa<int>({int(e.targets.size())});
            for(int i=0;i<int(e.targets.size());++i){std::copy_n(e.sources[i].data(),10,sources.mutable_data()+i*10);targets.mutable_data()[i]=e.targets[i];}
            d["source_rows"]=sources;d["target_slots"]=targets;return d;
        })

        .def("control_scores",[](WaveDiag &b,const SmallPlayoutNet &net){
            int n=b.active.size(),w=0;for(auto i:b.active)w=std::max(w,int(b.lanes[i].obs.actions.size()));auto scores=wa<float>({n,w});auto p=scores.mutable_data();
            {py::gil_scoped_release release;auto lock=b.pool.acquire();b.pool.runIndependent(n,[&](std::size_t i){
                auto vals=net.logits(b.lanes[b.active[i]].obs);std::copy(vals.begin(),vals.end(),p+i*w);});}
            return scores;
        })
        .def("results",[](WaveDiag &b){
            if(b.awaiting||b.nextTask.load()<int(b.local.size()))throw std::logic_error("unfinished worldline jobs");
            std::vector<SmallSearchResult> rows(b.roots.size());
            for(int i=0;i<int(rows.size());++i){auto &v=rows[i];auto &r=*b.roots[i];v.features=smallFeatures(r.root);auto n=r.actions.size();v.visits.resize(n);v.terminalVisits.resize(n);v.values.resize(n);v.stats.turnCardLimitDecisions=smallTurnCardLimitReached(r.root);}
            for(int i=0;i<int(b.local.size());++i){auto &v=rows[b.workRoot[i]];int j=b.chosen[i];++v.visits[j];if(std::isfinite(b.values[i])){v.values[j]+=b.values[i];++v.terminalVisits[j];}
                ++v.stats.simulations;v.stats.playoutTransitions+=b.depths[i]+1;v.stats.forcedPolicySkips+=b.forceN[i];v.stats.policyCalls+=b.depths[i]-b.forceN[i];v.stats.actionScores+=b.actionN[i];
                ++v.stats.playoutLengths[b.depths[i]];const auto outcome=sts::Outcome(b.outcomes[i]);if(outcome==sts::Outcome::PLAYER_VICTORY)++v.stats.terminalWins;else if(outcome==sts::Outcome::PLAYER_LOSS)++v.stats.terminalLosses;else if(outcome==sts::Outcome::PLAYER_ESCAPE)++v.stats.terminalEscapes;else ++v.stats.depthCuts;}
            py::list output;for(auto &v:rows){for(int j=0;j<int(v.values.size());++j)if(v.visits[j])v.values[j]/=v.visits[j];v.action=v.features.actions.front().bits;output.append(result(v));}return output;
        })

        .def("enable_mapped",&WaveDiag::enableMapped).def("reset_groups",&WaveDiag::resetGroups).def_readwrite("zone_mode",&WaveDiag::zoneMode).def_readwrite("zone_resident",&WaveDiag::resident).def_readwrite("short_atoms",&WaveDiag::shortAtoms).def_readwrite("aos_mode",&WaveDiag::aosMode)
        .def("produce",[](WaveDiag &b,py::list fields,int tokenCap,int grain){
            if(!b.aosMode||fields.size()!=14)throw std::logic_error("bad compact packet");
            auto get=[&](int i){return fields[i].cast<py::array>();};auto ip=[&](int i){return static_cast<int*>(get(i).mutable_data());};
            auto st=static_cast<float*>(get(0).mutable_data()),ac=static_cast<float*>(get(1).mutable_data());auto ky=ip(2),ow=ip(3),qs=ip(4),sc=ip(5),tg=ip(6),aos=ip(7),eid=ip(8),ex=ip(9),rootid=ip(10),offp=ip(11),lenp=ip(12);auto rnd=static_cast<double*>(get(13).mutable_data());
            const int B=get(0).shape(0),Q=get(4).shape(1),N=get(1).shape(0),E=get(7).shape(0),nr=b.active.size();

            if(!b.gpuChoice||b.awaiting||grain<1||grain>64)throw std::logic_error("bad fused phase");
            const int lanes=b.lanes.size();
            std::atomic<int> nextRow{0},nextAction{0},nextEntity{0},maxT{0},maxQ{0};
            int actual=0,acts=0,ents=0,mt=0,mq=0;
            {
              py::gil_scoped_release release;auto lock=b.pool.acquire();b.active.resize(lanes);b.offset.resize(lanes);b.entityOffsets.resize(lanes);
              b.pool.runIndependent((lanes+grain-1)/grain,[&](std::size_t block){
                 int begin=block*grain,stop=std::min(lanes,begin+grain), rows=0,actions=0,entities=0,bt=0,bq=0;
                 for(int k=begin;k<stop;++k){
                    b.prepareLane(k);auto &l=b.lanes[k];if(!l.ready)continue;
                    ++rows;actions+=l.obs.actions.size();entities+=l.entity.entities.size();bt=std::max(bt,int(l.live.size()));bq=std::max(bq,int(l.queries.size()));
                 }
                 if(!rows)return;
                 int i=nextRow.fetch_add(rows,std::memory_order_relaxed),o=nextAction.fetch_add(actions,std::memory_order_relaxed),e=nextEntity.fetch_add(entities,std::memory_order_relaxed);
                 auto mx=[](std::atomic<int>&v,int n){int cur=v.load(std::memory_order_relaxed);while(n>cur&&!v.compare_exchange_weak(cur,n,std::memory_order_relaxed)){};};
                 mx(maxT,bt);mx(maxQ,bq);
                 for(int k=begin;k<stop;++k){auto &l=b.lanes[k];if(!l.ready)continue;int a=l.obs.actions.size(),n=l.entity.entities.size();
                     b.active[i]=k;l.lastRow=i;b.offset[i]=o;b.entityOffsets[i]=e;
                     if(i<B && o+a<=N && e+n<=E && int(l.queries.size())<=Q && int(l.live.size())<=tokenCap){
                offp[i]=o;lenp[i]=a;rnd[i]=l.drawNext;rootid[i]=b.roots[b.workRoot[l.task]]->keyForInventory;
                std::copy_n(l.obs.state.data(),32,st+i*32);std::copy(l.queries.begin(),l.queries.end(),qs+i*Q);
                std::memcpy(aos+e*19,l.entity.entities.data(),n*sizeof(PlayoutEntity));std::fill_n(eid+e,n,i);
                if(l.rawExtra.empty())std::fill_n(ex+e,n,-1);else std::copy_n(l.rawExtra.data(),n,ex+e);
                for(int j=0;j<a;++j){auto &f=l.obs.actions[j];ow[o+j]=i;std::copy_n(f.x.data(),24,ac+(o+j)*24);
                    std::fill_n(ky+(o+j)*10,10,-1);std::copy_n(f.keys.data(),f.keyCount,ky+(o+j)*10);std::copy_n(l.src[j].data(),10,sc+(o+j)*10);tg[o+j]=l.target[j];}
                     }
                     ++i;o+=a;e+=n;
                 }
              });
              actual=nextRow.load();acts=nextAction.load();ents=nextEntity.load();mt=maxT.load();mq=maxQ.load();
              b.active.resize(actual);b.offset.resize(actual);b.entityOffsets.resize(actual);b.awaiting=actual>0;
              if(ents<=E)std::fill(eid+ents,eid+E,-1);if(actual<=B)std::fill(lenp+actual,lenp+B,0);
            }
            return py::make_tuple(actual,acts,mt,mq,ents,actual<=B && acts<=N && mt<=tokenCap && mq<=Q && ents<=E);
        })
        .def("pack_aos",[](WaveDiag &b,py::list fields){
            if(!b.aosMode||fields.size()!=14)throw std::logic_error("bad compact packet");
            auto get=[&](int i){return fields[i].cast<py::array>();};auto ip=[&](int i){return static_cast<int*>(get(i).mutable_data());};
            auto st=static_cast<float*>(get(0).mutable_data()),ac=static_cast<float*>(get(1).mutable_data());auto ky=ip(2),ow=ip(3),qs=ip(4),sc=ip(5),tg=ip(6),aos=ip(7),eid=ip(8),ex=ip(9),rootid=ip(10),offp=ip(11),lenp=ip(12);auto rnd=static_cast<double*>(get(13).mutable_data());
            const int B=get(0).shape(0),Q=get(4).shape(1),N=get(1).shape(0),E=get(7).shape(0),nr=b.active.size();
            if(nr>B||!nr||int(b.entityOffsets.size())!=nr)throw std::logic_error("packet rank");
            static_assert(sizeof(PlayoutEntity)==19*sizeof(int),"unexpected public row ABI");
            static_assert(offsetof(PlayoutEntity,numbers)==3*sizeof(int),"unexpected numeric offset");
            py::gil_scoped_release release;auto lock=b.pool.acquire();int actual=b.entityOffsets.back()+b.lanes[b.active.back()].entity.entities.size();
            if(actual>E)throw std::logic_error("too many raw rows");std::fill(eid+actual,eid+E,-1);std::fill(lenp+nr,lenp+B,0);
            constexpr int grain=8;b.pool.runIndependent((nr+grain-1)/grain,[&](std::size_t group){
              for(int i=group*grain;i<std::min(nr,int(group+1)*grain);++i){auto &l=b.lanes[b.active[i]];int o=b.offset[i],e=b.entityOffsets[i],a=l.obs.actions.size(),n=l.entity.entities.size();
                if(o+a>N||int(l.queries.size())>Q)throw std::logic_error("small compact packet");
                offp[i]=o;lenp[i]=a;rnd[i]=l.drawNext;rootid[i]=b.roots[b.workRoot[l.task]]->keyForInventory;
                std::copy_n(l.obs.state.data(),32,st+i*32);std::copy(l.queries.begin(),l.queries.end(),qs+i*Q);
                std::memcpy(aos+e*19,l.entity.entities.data(),n*sizeof(PlayoutEntity));std::fill_n(eid+e,n,i);
                if(l.rawExtra.empty())std::fill_n(ex+e,n,-1);else std::copy_n(l.rawExtra.data(),n,ex+e);
                for(int j=0;j<a;++j){auto &f=l.obs.actions[j];ow[o+j]=i;std::copy_n(f.x.data(),24,ac+(o+j)*24);
                    std::fill_n(ky+(o+j)*10,10,-1);std::copy_n(f.keys.data(),f.keyCount,ky+(o+j)*10);std::copy_n(l.src[j].data(),10,sc+(o+j)*10);tg[o+j]=l.target[j];}
              }
            });
        })

        .def_readwrite("gpu_choice",&WaveDiag::gpuChoice)
        .def("choose",[](WaveDiag &b,py::array_t<int,py::array::c_style> picks){
            if(!b.awaiting||!b.gpuChoice||picks.ndim()!=1||picks.size()<int(b.active.size()))throw std::logic_error("invalid GPU choices");
            auto ptr=picks.data();b.pendingPicks.assign(ptr,ptr+b.active.size());b.awaiting=false;
        })
        .def_readwrite("raw_mode",&WaveDiag::rawMode)
        .def("entity_count",[](WaveDiag &b){int n=0;b.entityOffsets.resize(b.active.size());for(int i=0;i<int(b.active.size());++i){b.entityOffsets[i]=n;n+=b.lanes[b.active[i]].entity.entities.size();}return n;})
        .def("pack_raw",[](WaveDiag &b,py::list fields){
            if(!b.rawMode||!b.fullTokens||fields.size()!=(b.gpuChoice?18:15))throw std::logic_error("raw form requires the full learned policy");
            auto get=[&](int i){return fields[i].cast<py::array>();};const int B=get(0).shape(0),T=get(4).shape(1),Q=get(8).shape(1),N=get(1).shape(0),E=get(11).shape(0);
            if(b.entityOffsets.size()!=b.active.size()||B<int(b.active.size()))throw std::logic_error("no raw shape");
            auto fp=[&](int i){return static_cast<float*>(get(i).mutable_data());};auto ip=[&](int i){return static_cast<int*>(get(i).mutable_data());};
            auto st=fp(0),ac=fp(1),nums=fp(11);auto ky=ip(2),ow=ip(3),cnt=ip(4),sy=ip(5),id=ip(6),qs=ip(8),sc=ip(9),tg=ip(10),key=ip(12),dest=ip(13),ex=ip(14);auto pr=static_cast<bool*>(get(7).mutable_data());
            auto offptr=b.gpuChoice?ip(15):nullptr;auto lenptr=b.gpuChoice?ip(16):nullptr;double *rnd=b.gpuChoice?static_cast<double*>(get(17).mutable_data()):nullptr;
            py::gil_scoped_release release;auto lock=b.pool.acquire();
            const int erows=b.entityOffsets.back()+b.lanes[b.active.back()].entity.entities.size();
            if(erows>E)throw std::logic_error("insufficient entity capacity");std::fill(dest+erows,dest+E,-1);
            const int grain=8,nrows=b.active.size();
            b.pool.runIndependent((nrows+grain-1)/grain,[&](std::size_t block){
             for(int i=block*grain;i<std::min(nrows,int(block+1)*grain);++i){
                auto &l=b.lanes[b.active[i]];int m=l.obs.actions.size(),off=b.offset[i],oo=b.entityOffsets[i],tok=l.live.size(),query=l.queries.size();
                if(tok>T||query>Q||off+m>N)throw std::logic_error("small raw buffers");if(b.gpuChoice){offptr[i]=off;lenptr[i]=m;rnd[i]=l.drawNext;}
                std::copy_n(l.obs.state.data(),32,st+i*32);auto p=i*T;
                for(int j=0;j<tok;++j){int k=l.live[j];id[p+j]=std::min(k,33);sy[p+j]=l.symbols[k];pr[p+j]=true;cnt[p+j]=l.rawCounts[k];}
                std::fill_n(pr+p+tok,T-tok,false);std::fill_n(cnt+p+tok,T-tok,0);std::copy_n(l.queries.data(),query,qs+i*Q);
                for(int j=0;j<m;++j){auto &f=l.obs.actions[j];ow[off+j]=i;std::copy_n(f.x.data(),24,ac+(off+j)*24);std::fill_n(ky+(off+j)*10,10,-1);std::copy_n(f.keys.data(),f.keyCount,ky+(off+j)*10);std::copy_n(l.src[j].data(),10,sc+(off+j)*10);tg[off+j]=l.target[j];}
                for(int j=0;j<int(l.entity.entities.size());++j){auto &v=l.entity.entities[j];key[oo+j]=v.id*15+v.zone;dest[oo+j]=p+l.rawOwner[j];ex[oo+j]=l.rawExtra[j]<0?-1:p+l.rawExtra[j];std::copy_n(v.numbers.data(),16,nums+(oo+j)*16);}
             }
            });
        })

        .def_readwrite("fast_codec",&WaveDiag::fastCodec)
        .def_readwrite("full_tokens",&WaveDiag::fullTokens)
        .def("ready",[](WaveDiag &b){ {py::gil_scoped_release g;b.prepare();} int acts=0,t=0,q=0;
            b.offset.resize(b.active.size());for(int i=0;i<int(b.active.size());++i){
                auto &l=b.lanes[b.active[i]];b.offset[i]=acts;acts+=l.obs.actions.size();t=std::max(t,int(l.live.size()));q=std::max(q,int(l.queries.size()));}
            return py::make_tuple(b.active.size(),acts,t,q); })
        .def("pack_pinned",[](WaveDiag &b,py::list fields){
            if(!b.fullTokens||fields.size()!=11)throw std::logic_error("expected full learned policy");
            auto get=[&](int i){return fields[i].cast<py::array>();};auto shape=get(4);const int n=shape.shape(0),t=shape.shape(1),q=get(8).shape(1),na=get(1).shape(0);
            if(n<int(b.active.size())||t<1||q<1)throw std::logic_error("bad buffer size");
            auto fp=[&](int i){return static_cast<float*>(get(i).mutable_data());};auto ip=[&](int i){return static_cast<int*>(get(i).mutable_data());};
            auto st=fp(0),ac=fp(1),ft=fp(4);auto keys=ip(2),owner=ip(3),sy=ip(5),id=ip(6),qs=ip(8),sc=ip(9),tg=ip(10);auto pr=static_cast<bool*>(get(7).mutable_data());
            py::gil_scoped_release release;auto lock=b.pool.acquire();
            b.pool.runIndependent(b.active.size(),[&](std::size_t i){auto &l=b.lanes[b.active[i]];int m=l.obs.actions.size(),off=b.offset[i];int tok=l.live.size(),query=l.queries.size();
                if(tok>t||query>q||off+m>na)throw std::logic_error("insufficient buffer");
                std::copy_n(l.obs.state.data(),32,st+i*32);auto p=i*t;
                for(int j=0;j<tok;++j){int k=l.live[j];id[p+j]=std::min(k,33);sy[p+j]=l.symbols[k];pr[p+j]=true;const auto &f=(b.fastCodec&&b.fullTokens&&l.features.size()==33)?l.cached->cachedTokens[k]:l.features[k];std::copy_n(f.data(),128,ft+(p+j)*128);}
                std::fill_n(pr+p+tok,t-tok,false);std::copy(l.queries.begin(),l.queries.end(),qs+i*q);
                for(int j=0;j<m;++j){const auto &f=l.obs.actions[j];owner[off+j]=i;std::copy_n(f.x.data(),24,ac+(off+j)*24);
                  std::fill_n(keys+(off+j)*10,10,-1);std::copy_n(f.keys.data(),f.keyCount,keys+(off+j)*10);
                  std::copy_n(l.src[j].data(),10,sc+(off+j)*10);tg[off+j]=l.target[j];}
            });
        })
        .def("advance_flat",[](WaveDiag &b,py::array_t<float,py::array::c_style> x){
            if(x.ndim()!=1||b.active.empty()||x.size()<b.offset.back()+int(b.lanes[b.active.back()].obs.actions.size()))throw std::logic_error("bad flat logits");
            const auto p=x.data();py::gil_scoped_release g;b.advance(p,b.active.size(),-1);
        })

        .def("summary",[](WaveDiag &b){
            std::uint64_t steps=0,forced=0,scored=0;for(auto &l:b.lanes){steps+=l.steps;forced+=l.forced;scored+=l.scored;}
            int finished=0;double sum=0;for(double v:b.values)if(std::isfinite(v)){++finished;sum+=v;}
            py::dict d;d["roots"]=b.roots.size();d["simulations"]=b.local.size();d["terminal"]=finished;d["transitions"]=steps;d["forced"]=forced;d["scored"]=scored;d["return_sum"]=sum;double ps=0,es=0;std::uint64_t dirty=0,all=0;for(auto &l:b.lanes){ps+=l.pubSec;es+=l.encodeSec;dirty+=l.dirtyTokens;all+=l.allTokens;}d["public_thread_seconds"]=ps;d["encode_thread_seconds"]=es;d["dirty_tokens"]=dirty;d["all_tokens"]=all;return d;
        });
}
}

#ifndef STS_FULL480_STANDALONE
#include <numeric>
namespace sts::search { int writeNormalFacts(const BattleContext&,int,int*,float*,int); }
#include "raw_public.h"
namespace {
static_assert(std::is_trivially_copyable_v<sts::CardInstance> && sizeof(sts::CardInstance)==20);
static_assert(offsetof(sts::CardInstance,id)==0 && offsetof(sts::CardInstance,uniqueId)==2 && offsetof(sts::CardInstance,specialData)==4);
static_assert(offsetof(sts::CardInstance,cachedBaseDamage)==8 && offsetof(sts::CardInstance,cost)==12 && offsetof(sts::CardInstance,costForTurn)==13);
static_assert(offsetof(sts::CardInstance,upgraded)==14 && offsetof(sts::CardInstance,freeToPlayOnce)==15 && offsetof(sts::CardInstance,retain)==16 && offsetof(sts::CardInstance,exhaust)==17);
static int cardRows(const WaveLane &l){
    if(l.state->inputState!=sts::InputState::PLAYER_NORMAL)return 0;
    const auto &c=l.state->cards;return c.cardsInHand+c.drawPile.size()+c.discardPile.size()+c.exhaustPile.size()+2;
}
struct HostPacket {
    float *states,*actions; int *keys,*owner,*queries,*source,*target,*aos,*eid,*extra,*root,*offset,*length;
    double *random; int B,Q,N,E,T;bool shortAtoms; int *cards,*zone;float *dense;int C;bool rawPublic;
    HostPacket(py::list fields,int t):T(t) {
        if(fields.size()!=17)throw std::logic_error("wrong field count");
        auto at=[&](int i){return fields[i].cast<py::array>();};
        auto ip=[&](int i){return static_cast<int*>(at(i).mutable_data());};
        cards=ip(14);zone=ip(15);dense=static_cast<float*>(at(16).mutable_data());C=at(14).shape(0);rawPublic=at(16).ndim()==2;if(rawPublic&&at(16).shape(1)!=RAW_PUBLIC_WORDS)throw std::logic_error("raw public ABI shape");shortAtoms=at(2).shape(1)==1;states=static_cast<float*>(at(0).mutable_data());actions=static_cast<float*>(at(1).mutable_data());
        keys=ip(2);owner=ip(3);queries=ip(4);source=ip(5);target=ip(6);aos=ip(7);eid=ip(8);extra=ip(9);root=ip(10);offset=ip(11);length=ip(12);
        random=static_cast<double*>(at(13).mutable_data());B=at(0).shape(0);Q=at(4).shape(1);N=at(1).shape(0);E=at(7).shape(0);
    }
    void put(WaveDiag &b, WaveLane &l, int i,int o,int e,int cardOff) const{
        const int a=l.obs.actions.size(),n=shortAtoms?l.factBound:int(l.entity.entities.size());
        const int cardCount=cardRows(l);if(i>=B||o+a>N||e+n>E||cardOff+cardCount>C||int(l.queries.size())>Q||int(l.live.size())>T)return;
        offset[i]=o;length[i]=a;random[i]=l.drawNext;root[i]=b.roots[b.workRoot[l.task]]->keyForInventory;if(b.resident){int lane=&l-b.lanes.data();if(root[i]<0||root[i]>=4096||lane>=8192)throw std::logic_error("invalid cache address");root[i]|=(lane<<12)|((l.cardCache?l.cardCache->reuseMask:0)<<25);}
        std::memcpy(states+i*32,l.obs.state.data(),sizeof(float)*32);std::copy(l.queries.begin(),l.queries.end(),queries+i*Q);std::fill(queries+i*Q+l.queries.size(),queries+(i+1)*Q,0);
        if(shortAtoms){
            if(rawPublic)writeRawPublic(*l.state,i,e,n,reinterpret_cast<int*>(dense)+i*RAW_PUBLIC_WORDS,aos+e*5,rawPublicVerify,b.mappedAddress(l));
            else {const int c=writeNormalFacts(*l.state,i,aos+e*5,dense+i*7*16,n);
                for(int j=c;j<n;++j)aos[(e+j)*5]=-1;}
        }else{std::memcpy(aos+e*19,l.entity.entities.data(),n*sizeof(PlayoutEntity));std::fill_n(eid+e,n,i);
        if(l.rawExtra.empty())std::fill_n(extra+e,n,-1);else std::copy_n(l.rawExtra.data(),n,extra+e);}
        auto *h=zone+i*8;
        if(shortAtoms){
            const auto &bc=*l.state;const auto &c=bc.cards;
            h[0]=cardOff;h[1]=c.cardsInHand;h[2]=c.drawPile.size();
            h[3]=bc.player.hasRelic<sts::RelicId::FROZEN_EYE>()?h[2]:std::clamp(c.knownDrawTop,0,h[2]);
            h[4]=c.discardPile.size();h[5]=c.exhaustPile.size();h[6]=b.roots[b.workRoot[l.task]]->deck;h[7]=cardCount;
            auto *dst=cards+size_t(cardOff)*5;
            auto copy=[&](const auto& pile,int count){if(count){std::memcpy(dst,&pile[0],sizeof(sts::CardInstance)*count);dst+=count*5;}};
            copy(c.hand,h[1]);copy(c.drawPile,h[2]);copy(c.discardPile,h[4]);copy(c.exhaustPile,h[5]);copy(c.stasisCards,2);
        }else{std::fill_n(h,8,0);}
        if(shortAtoms){
            if(l.state->inputState!=sts::InputState::PLAYER_NORMAL)throw std::logic_error("non-normal action sent to fixed decoder");
            auto &bc=*l.state;
            for(int j=0;j<a;++j){Action action(l.obs.actions[j].bits);auto kind=action.getActionType();
                auto *f=reinterpret_cast<int*>(actions)+(o+j)*13;
                int key=0,cost=0,up=0,type=0,special=0,flags=0;
                if(kind==ActionType::CARD){
                    auto &c=bc.cards.hand[action.getSourceIdx()];up=c.getUpgradeCount();key=3+2*(int(c.id)-1)+(up>0);
                    cost=c.costForTurn;type=int(c.getType());special=c.specialData;
                    flags=int(c.freeToPlayOnce)|(int(c.doesExhaust())<<1)|(int(c.retain)<<2)|(int(c.isEthereal())<<3);
                }else if(kind==ActionType::POTION){key=3+2*int(sts::CardId::ZAP)+int(bc.potions[action.getSourceIdx()])-int(sts::Potion::AMBROSIA);}
                else if(kind!=ActionType::END_TURN)throw std::logic_error("unknown normal action");
                f[0]=int(kind);f[1]=cost;f[2]=up;f[3]=type;f[4]=special;f[5]=flags;
                if(l.target[j]>=0){int t=action.getTargetIdx();auto &m=bc.monsters.arr[t];f[6]=t+1;f[7]=m.curHp;f[8]=m.maxHp;f[9]=m.block;f[10]=m.strength;f[11]=m.vulnerable;f[12]=m.weak;}
                else std::fill_n(f+6,7,0);
                keys[o+j]=key;owner[o+j]=i;source[o+j]=l.src[j][0];target[o+j]=l.target[j];
            }
            return;
        }
        for(int j=0;j<a;++j){auto &f=l.obs.actions[j];owner[o+j]=i;std::copy_n(f.x.data(),24,actions+(o+j)*24);
            std::fill_n(keys+(o+j)*10,10,-1);std::copy_n(f.keys.data(),f.keyCount,keys+(o+j)*10);std::copy_n(l.src[j].data(),10,source+(o+j)*10);target[o+j]=l.target[j];}
    }
    void pad(int b,int e) const {if(shortAtoms)eid[0]=std::min(e,E);else if(e<=E)std::fill(eid+e,eid+E,-1);if(b<=B){std::fill(length+b,length+B,0);std::fill(zone+b*8,zone+B*8,0);}}
};
struct SplitDiag {
    WaveDiag *w;
    struct Meta{int b=0,a=0,e=0,t=0,q=0,c=0;};
    struct Lane{int group=0,row=0,action=0,entity=0,card=0;};
    std::vector<Lane> layout;
    // Completed lanes cannot receive additional tasks in this fixed root group.
    std::vector<int> blocks;
    std::vector<uint64_t> mask;
    int laneGrain=0, lastRows=-1;
    bool compact=std::getenv("TAIL_COMPACT") && std::atoi(std::getenv("TAIL_COMPACT"));
    int mergeBelow=std::getenv("TAIL_MERGE")?std::atoi(std::getenv("TAIL_MERGE")):0;
    std::array<std::vector<int>,3> inverse;
    std::array<Meta,3> meta;
    SplitDiag(WaveDiag &ref):w(&ref),layout(ref.lanes.size()){auto *base=w;
        for(auto &v:inverse)v.resize(layout.size());
        base->active.resize(layout.size());std::iota(base->active.begin(),base->active.end(),0);
    }
    std::vector<std::array<int,7>> produce(py::list lo,py::list hi,py::list extra,int lt,int ht,int et,int threshold,int grain){
        if(w->awaiting||grain<1||grain>64)throw std::logic_error("invalid split phase");
        HostPacket packs[]={HostPacket(lo,lt),HostPacket(hi,ht),HostPacket(extra,et)};
        struct Counts {std::atomic<int> b{0},a{0},e{0},t{0},q{0},c{0};}; Counts sum[3];
        if(!laneGrain){
            laneGrain=grain;int n=(int(layout.size())+grain-1)/grain;
            blocks.resize(n);std::iota(blocks.begin(),blocks.end(),0);mask.resize(n);
            for(int k=0;k<n;++k){int width=std::min(grain,int(layout.size())-k*grain);
                mask[k]=width==64?~uint64_t(0):((uint64_t(1)<<width)-1);}
        }
        if(grain!=laneGrain)throw std::logic_error("grain changed on active pool");
        int prev=lastRows<0?int(layout.size()):lastRows;
        if(mergeBelow>0 && prev<=mergeBelow) threshold=0;
        {py::gil_scoped_release release;auto lock=w->pool.acquire();
        w->pool.runIndependent(blocks.size(),[&](std::size_t pos){int chunk=blocks[pos];
            int begin=chunk*grain,end=std::min(begin+grain,int(layout.size()));Meta local[3];
            uint64_t alive=0,candidates=mask[chunk];
            while(candidates){int shift=__builtin_ctzll(candidates);candidates&=candidates-1;int k=begin+shift;
                w->prepareLane(k);auto &s=w->lanes[k];if(!s.ready)continue;alive|=uint64_t(1)<<shift;
                int g=(s.state->inputState==sts::InputState::PLAYER_NORMAL)?1:2;
                if (s.state->inputState==sts::InputState::PLAYER_NORMAL && s.live.size()==33) {
                    const auto &bc=*s.state;const auto &c=bc.cards;
                    const int draw=int(c.drawPile.size());
                    const int known=bc.player.hasRelic<sts::RelicId::FROZEN_EYE>()?draw:std::clamp(c.knownDrawTop,0,draw);
                    const int upper=11+(known<draw)+(known>0)+(!c.discardPile.empty())+(!c.exhaustPile.empty())
                        +(c.stasisCards[0].id!=sts::CardId::INVALID||c.stasisCards[1].id!=sts::CardId::INVALID)
                        +bc.potionCapacity+c.cardsInHand;
                    g=upper>threshold;
                }
                layout[k].group=g;auto &v=local[g];
                ++v.b;v.c+=cardRows(s);v.a+=s.obs.actions.size();v.e+=g<2?s.factBound:int(s.entity.entities.size());v.q=std::max(v.q,int(s.queries.size()));v.t=std::max(v.t,int(s.live.size()));
            }
            int rows[3]={},acts[3]={},ents[3]={},cards[3]={};
            auto mx=[](std::atomic<int>&v,int n){int cur=v.load();while(n>cur&&!v.compare_exchange_weak(cur,n)){};};
            for(int g=0;g<3;++g){auto &v=local[g];auto &s=sum[g];if(v.b){rows[g]=s.b.fetch_add(v.b);acts[g]=s.a.fetch_add(v.a);ents[g]=s.e.fetch_add(v.e);cards[g]=s.c.fetch_add(v.c);mx(s.t,v.t);mx(s.q,v.q);}}
            if(compact)mask[chunk]=alive;
            while(alive){int k=begin+__builtin_ctzll(alive);alive&=alive-1;auto &s=w->lanes[k];auto &v=layout[k];int g=v.group;
                v.row=rows[g]++;v.action=acts[g];acts[g]+=s.obs.actions.size();v.entity=ents[g];ents[g]+=g<2?s.factBound:int(s.entity.entities.size());
                v.card=cards[g];cards[g]+=cardRows(s);inverse[g][v.row]=k;s.lastRow=k;packs[g].put(*w,s,v.row,v.action,v.entity,v.card);
            }
        });
        lastRows=sum[0].b+sum[1].b+sum[2].b;
        if(compact && lastRows<int(layout.size()))blocks.erase(std::remove_if(blocks.begin(),blocks.end(),[&](int k){return mask[k]==0;}),blocks.end());
        for(int g=0;g<3;++g){auto &s=sum[g];meta[g]={s.b.load(),s.a.load(),s.e.load(),s.t.load(),s.q.load(),s.c.load()};packs[g].pad(meta[g].b,meta[g].e);}
        w->awaiting=meta[0].b+meta[1].b+meta[2].b>0;}
        std::vector<std::array<int,7>> out;
        for(int g=0;g<3;++g){auto &v=meta[g];auto &p=packs[g];out.push_back({v.b,v.a,v.t,v.q,v.e,v.c,v.c<=p.C&&v.b<=p.B&&v.a<=p.N&&v.e<=p.E&&v.t<=p.T&&v.q<=p.Q});}return out;
    }
    void pack(int g,py::list fields,int tokens){
        if(g<0||g>2)throw std::logic_error("invalid group"); HostPacket p(fields,tokens);auto v=meta[g];
        if(v.c>p.C||v.b>p.B||v.a>p.N||v.e>p.E||v.t>p.T||v.q>p.Q)throw std::logic_error("small repack");
        py::gil_scoped_release release;auto lock=w->pool.acquire();p.pad(v.b,v.e);
        w->pool.runIndependent((v.b+7)/8,[&](size_t block){for(int j=block*8;j<std::min(v.b,int(block*8)+8);++j){int k=inverse[g][j];auto &x=layout[k];p.put(*w,w->lanes[k],j,x.action,x.entity,x.card);}});
    }
    void choose(py::array_t<int,py::array::c_style> lo,py::array_t<int,py::array::c_style> hi,py::array_t<int,py::array::c_style> extra){
        if(!w->awaiting||lo.size()<meta[0].b||hi.size()<meta[1].b||extra.size()<meta[2].b)throw std::logic_error("missing choices");
        const int *p[]={lo.data(),hi.data(),extra.data()};w->pendingPicks.resize(layout.size());
        for(int g=0;g<3;++g)for(int i=0;i<meta[g].b;++i)w->pendingPicks[inverse[g][i]]=p[g][i];w->awaiting=false;
    }
};
}

#include <cuda.h>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>

// Graphs belong to Python stages. Each Slot keeps a strong stage reference until its CUDA event completes.
namespace {
using HostClock=std::chrono::steady_clock;
double seconds(HostClock::time_point at){return std::chrono::duration<double>(HostClock::now()-at).count();}
void cudaCheck(CUresult status){if(status!=CUDA_SUCCESS){const char *text=nullptr;cuGetErrorString(status,&text);throw std::runtime_error(text?text:"CUDA driver failure");}}
void waitReady(CUevent event,int pipe,int group){
    auto start=HostClock::now();CUresult rc;
    while((rc=cuEventQuery(event))==CUDA_ERROR_NOT_READY){
        double elapsed=seconds(start);
        if(elapsed>25){std::fprintf(stderr,"CUDA_PACKET_STALL pipe=%d group=%d\n",pipe,group);std::fflush(stderr);std::_Exit(86);}
        if(elapsed>.001)std::this_thread::sleep_for(std::chrono::microseconds(15));
    }
    cudaCheck(rc);
}
struct PacketSlot {
    py::object owner;
    py::list host;
    CUgraphExec graph=nullptr;
    int *picks=nullptr;
    struct Copy {CUdeviceptr device;const void *host;std::size_t bytes;};
    std::array<Copy,3> inputs{};
    struct PartialCopy {CUdeviceptr device;const void *host;std::size_t capacity,rowBytes;int limit;};
    std::vector<PartialCopy> partials;
    int copyMode=std::getenv("PARTIAL_UPLOAD")?std::atoi(std::getenv("PARTIAL_UPLOAD")):0;
    int entities=0,cards=0;
    unsigned long long uploadedBytes=0,uploadCalls=0;
    CUdeviceptr selected=0;
    void submit(CUstream stream){
        if(copyMode<0||copyMode>4)throw std::logic_error("unknown upload mode");
        if(copyMode>=3){
            const int limits[]={valid,count,entities,cards};
            std::array<CUdeviceptr,17> dst{},src{};
            std::array<std::size_t,17> size{};std::size_t n=0;
            for(const auto &m:partials){
                std::size_t bytes=copyMode==4 && m.limit>=0 ? std::size_t(limits[m.limit])*m.rowBytes : m.capacity;
                if(bytes>m.capacity)throw std::logic_error("batch upload exceeds captured input");
                if(bytes){dst[n]=m.device;src[n]=reinterpret_cast<CUdeviceptr>(m.host);size[n]=bytes;++n;uploadedBytes+=bytes;}
            }
            if(n){
                CUmemcpyAttributes attrs{};attrs.srcAccessOrder=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
                std::size_t first=0,failed=std::size_t(-1);
                CUresult rc=cuMemcpyBatchAsync(dst.data(),src.data(),size.data(),n,&attrs,&first,1,&failed,stream);
                if(rc!=CUDA_SUCCESS){std::fprintf(stderr,"BATCH_COPY_ERROR mode=%d index=%zu count=%zu\n",copyMode,failed,n);cudaCheck(rc);}
                ++uploadCalls;
            }
        }else if(copyMode){
            const int limits[]={valid,count,entities,cards};
            for(const auto &m:partials){
                std::size_t bytes=copyMode==2 && m.limit>=0 ? std::size_t(limits[m.limit])*m.rowBytes : m.capacity;
                if(bytes>m.capacity)throw std::logic_error("partial upload exceeds captured input");
                if(bytes){cudaCheck(cuMemcpyHtoDAsync(m.device,m.host,bytes,stream));uploadedBytes+=bytes;++uploadCalls;}
            }
        }else for(auto m:inputs){cudaCheck(cuMemcpyHtoDAsync(m.device,m.host,m.bytes,stream));uploadedBytes+=m.bytes;++uploadCalls;}
        cudaCheck(cuGraphLaunch(graph,stream));
        cudaCheck(cuMemcpyDtoHAsync(picks,selected,valid*sizeof(int),stream));
    }
    int rows=0,tokens=1,valid=0,count=0;
    void set(py::object stage,bool isGraph){
        owner=std::move(stage);host=owner.attr("host").cast<py::list>();
        auto shape=owner.attr("shape").cast<py::tuple>();
        rows=shape[0].cast<int>();tokens=shape[1].cast<int>();
        if(isGraph){
            graph=reinterpret_cast<CUgraphExec>(owner.attr("graph").attr("raw_cuda_graph_exec")().cast<std::uintptr_t>());
            picks=static_cast<int*>(owner.attr("array").cast<py::array>().mutable_data());
            auto views=owner.attr("transfer").cast<py::list>();if(views.size()!=3)throw std::logic_error("three typed input slabs required");
            for(int i=0;i<3;++i){auto v=views[i].cast<py::tuple>();inputs[i]={v[0].cast<CUdeviceptr>(),reinterpret_cast<const void*>(v[1].cast<std::uintptr_t>()),v[2].cast<std::size_t>()};}
            partials.clear();
            if(copyMode){
                for(auto item:owner.attr("partial_transfer").cast<py::list>()){
                    auto v=item.cast<py::tuple>();partials.push_back({v[0].cast<CUdeviceptr>(),reinterpret_cast<const void*>(v[1].cast<std::uintptr_t>()),v[2].cast<std::size_t>(),v[3].cast<std::size_t>(),v[4].cast<int>()});
                }
                if(partials.size()!=17)throw std::logic_error("incomplete partial transfer contract");
                for(std::size_t a=0;a<partials.size();++a){
                    const auto &u=partials[a];
                    if(u.limit < -1 || u.limit > 3)throw std::logic_error("invalid input limit");
                    if(copyMode>=3)for(std::size_t b=0;b<a;++b){
                        const auto &v=partials[b];
                        if(u.capacity&&v.capacity&&u.device<v.device+v.capacity&&v.device<u.device+u.capacity)
                            throw std::logic_error("batch copy destinations overlap");
                    }
                }
            }
            selected=owner.attr("selection").cast<CUdeviceptr>();
        }
    }
};
struct PipeState {
    std::array<PacketSlot,3> slots;
    std::array<CUevent,3> events{};
    std::array<CUstream,3> streams{};
    bool active=true,pending=false;
    ~PipeState(){for(auto event:events)if(event)cuEventDestroy(event);}
};
py::dict drivePackets(py::list splits,py::list initial,py::list streamIds,py::function get,
                      py::function check,py::function fallback,int interval){
    const int n=splits.size();std::vector<std::unique_ptr<PipeState>> ps;
    std::map<std::pair<int,int>,int> hist;
    double warm=0,produce=0,wait=0,launch=0,refill=0,validTime=0,choose=0;
    long long actions=0,packets=0,fallbackRows=0,repacks=0,checks=0;double error=0;
    for(int i=0;i<n;++i){
        auto s=std::make_unique<PipeState>();
        for(int g=0;g<3;++g){
            auto stage=initial[i].cast<py::list>()[g].cast<py::object>();
            s->slots[g].set(stage,py::hasattr(stage,"graph"));
            s->streams[g]=reinterpret_cast<CUstream>(streamIds[i].cast<py::list>()[g].cast<std::uintptr_t>());
            if((i||g) && s->streams[g]!=reinterpret_cast<CUstream>(streamIds[0].cast<py::list>()[0].cast<std::uintptr_t>()))throw std::logic_error("shared GPU slabs require one ordered stream");
            cudaCheck(cuEventCreate(&s->events[g],CU_EVENT_DISABLE_TIMING));
        }
        ps.push_back(std::move(s));
    }
    bool fixture=std::getenv("SHARE_FIXTURE")&&std::atoi(std::getenv("SHARE_FIXTURE"));int remaining=n;
    while(remaining)for(int i=0;i<n;++i){
        auto &p=*ps[i];if(!p.active)continue;auto &sp=splits[i].cast<SplitDiag &>();
        if(p.pending){
            auto ts=HostClock::now();
            {py::gil_scoped_release release;for(int g=0;g<3;++g)if(p.slots[g].valid)waitReady(p.events[g],i,g);}
            wait+=seconds(ts);ts=HostClock::now();
            for(int g=0;g<3;++g){
                auto &h=p.slots[g];
                if(fixture && h.valid && *std::min_element(h.picks,h.picks+h.valid)>=0){
                    std::vector<int> expected(h.picks,h.picks+h.valid);h.submit(p.streams[g]);cudaCheck(cuEventRecord(p.events[g],p.streams[g]));
                    waitReady(p.events[g],i,g);fallback(h.owner);if(!std::equal(expected.begin(),expected.end(),h.picks))throw std::logic_error("replayed fallback changed picks");
                    fixture=false;}
                if(h.valid&&*std::min_element(h.picks,h.picks+h.valid)<0){h.submit(p.streams[g]);cudaCheck(cuEventRecord(p.events[g],p.streams[g]));waitReady(p.events[g],i,g);fallback(h.owner);fallbackRows+=h.valid;}
            }
            // Original choose contract: bind each returned action to the same lane.
            auto &w=*sp.w;w.pendingPicks.resize(w.lanes.size());
            if(!w.awaiting)throw std::logic_error("missing pending group");
            for(int g=0;g<3;++g){
                auto &h=p.slots[g];if(h.valid!=sp.meta[g].b)throw std::logic_error("stale packet size");
                for(int k=0;k<h.valid;++k){if(h.picks[k]<0)throw std::logic_error("fallback did not score action");w.pendingPicks[sp.inverse[g][k]]=h.picks[k];}
            }
            w.awaiting=false;choose+=seconds(ts);
        }
        auto ts=HostClock::now();
        auto &a=p.slots;
        auto dims=sp.produce(a[0].host,a[1].host,a[2].host,a[0].tokens,a[1].tokens,a[2].tokens,21,8);
        produce+=seconds(ts);
        if(!(dims[0][0]+dims[1][0]+dims[2][0])){p.active=false;--remaining;continue;}
        p.pending=true;
        for(int g=0;g<3;++g){
            auto &h=a[g];auto d=dims[g];h.valid=d[0];h.count=d[1];h.entities=d[4];h.cards=d[5];
            if(!h.valid)continue;
            if(!d[6]||(h.rows>512&&d[0]*2<h.rows)){
                ts=HostClock::now();
                auto got=get(i,g,d[0],d[2],d[3],d[1],d[4],d[5],true).cast<py::tuple>();
                warm+=got[1].cast<double>();auto stage=got[0].cast<py::object>();
                bool changed=!stage.is(h.owner);
                if(changed)h.set(std::move(stage),true);
                if(changed||!d[6]){sp.pack(g,h.host,h.tokens);++repacks;}
                refill+=seconds(ts);
            }
            ts=HostClock::now();h.submit(p.streams[g]);cudaCheck(cuEventRecord(p.events[g],p.streams[g]));
            launch+=seconds(ts);++hist[{g,d[0]}];actions+=d[1];++packets;
            if(interval&&packets%interval==1){
                ts=HostClock::now();waitReady(p.events[g],i,g);
                auto out=check(i,g,h.owner,d[0],d[1]).cast<py::tuple>();checks+=out[0].cast<int>();error=std::max(error,out[1].cast<double>());validTime+=seconds(ts);
            }
        }
    }
    py::dict counts,hh,d;
    unsigned long long uploadedBytes=0,uploadCalls=0;
    for(auto &pipe:ps)for(auto &slot:pipe->slots){uploadedBytes+=slot.uploadedBytes;uploadCalls+=slot.uploadCalls;}
    counts["upload_bytes"]=uploadedBytes;counts["upload_calls"]=uploadCalls;

    counts["warm"]=warm;counts["produce"]=produce;counts["wait"]=wait;counts["choose"]=choose;counts["graph_submit"]=launch;
    counts["refill_including_setup"]=refill;counts["validation"]=validTime;counts["fallback_rows"]=fallbackRows;counts["repacks"]=repacks;
    counts["actions"]=actions;counts["packets"]=packets;
    for(auto const &v:hist)hh[py::str(std::to_string(v.first.first)+","+std::to_string(v.first.second))]=v.second;
    py::list final;for(auto &p:ps){py::list row;for(auto &slot:p->slots)row.append(slot.owner);final.append(row);}
    d["stages"]=final;d["costs"]=counts;d["histogram"]=hh;d["checks"]=checks;d["maxerr"]=error;return d;
}
}

void bindSmallPlayout(py::module_ &m) {
    m.def("raw_public_schema",&rawPublicSchema);
    m.def("mapped_state_schema",&mappedStateSchema);
    m.def("raw_public_stats",[](){py::dict d;d["verified_writes"]=rawPublicRows.load();d["fallback_writes"]=rawPublicFallbacks.load();return d;});
    m.def("raw_public_fixture",[](const sts::BattleContext &source,int scenario){
        using namespace sts;
        if(!rawPublicVerify)throw std::logic_error("raw fixture requires VERIFY_RAW_PUBLIC=1");
        auto *fixtureBlock=new MappedBattleBlock(1);
        auto *fixtureState=new(fixtureBlock->host) BattleContext(source);
        py::capsule fixtureOwner(fixtureBlock,[](void *v){auto *b=static_cast<MappedBattleBlock*>(v);static_cast<BattleContext*>(b->host)->~BattleContext();delete b;});
        auto &bc=*fixtureState;
        if(scenario==1)bc.player.setHasRelic<R::RUNIC_DOME>(true);
        if(scenario==2)bc.player.setHasRelic<R::RUNIC_DOME>(false);
        if(scenario==3)bc.player.bombs.resize(17,Player::Bomb{2,37});
        if(scenario==4)bc.onUseRelicOrder.resize(129,R::PEN_NIB);
        if(scenario==5){bc.monsters.arr[2]=bc.monsters.arr[0];bc.monsters.arr[2].idx=2;bc.monsters.arr[0]=Monster();}
        if(scenario==6){bc.player.dexterity=-3;bc.player.strength=0;bc.player.statusBits0|=std::uint64_t(1)<<int(PS::STRENGTH);}
        if(scenario==7||scenario==8){bc.onUseRelicOrder.assign(scenario==7?16:17,R::PEN_NIB);bc.onEndTurnRelicOrder.clear();bc.onPostDrawRelicOrder.clear();bc.player.endTurnPowerOrder.clear();}
        if(scenario==9)bc.player.bombs.assign(16,Player::Bomb{2,37});
        if(scenario==10){for(int i=1;i<=26;++i)if(!bc.player.statusMap.count(PS(i)))bc.player.statusMap[PS(i)]=2;}
        if(scenario==11){bc.player.happyFlowerCounter=-3;bc.player.devaFormEnergyPerTurn=1024;bc.player.cardsPlayedThisTurn=300;bc.player.lastTargetedMonster=-1;bc.player.gold=-12;bc.monsters.arr[0].strength=-11;bc.monsters.arr[0].thiefSlashCount=9;bc.monsters.arr[0].thiefGoldAmount=123;}
        const int cap=2+21+128+int(bc.player.bombs.size())+128+int(bc.onUseRelicOrder.size()+bc.onEndTurnRelicOrder.size()+bc.onPostDrawRelicOrder.size()+bc.player.endTurnPowerOrder.size())+bc.potionCapacity+5*69+8;
        py::array_t<int> raw(RAW_PUBLIC_WORDS),facts({cap,5});
        std::fill_n(raw.mutable_data(),RAW_PUBLIC_WORDS,0);std::fill_n(facts.mutable_data(),cap*5,0);
        writeRawPublic(bc,0,0,cap,raw.mutable_data(),facts.mutable_data(),true,fixtureBlock->device);
        py::dict out;out["raw"]=raw;out["facts"]=facts;out["scenario"]=scenario;out["owner"]=fixtureOwner;return out;
    },py::arg("battle"),py::arg("scenario")=0);

    m.def("drive_packets",&drivePackets);
    m.attr("span_searing_blow")=int(sts::CardId::SEARING_BLOW);m.attr("span_invalid_card")=int(sts::CardId::INVALID);
    py::class_<SplitDiag>(m,"SplitDiag")
        .def(py::init<WaveDiag &>(),py::keep_alive<1,2>()).def("produce",&SplitDiag::produce)
        .def("pack",&SplitDiag::pack).def("choose",&SplitDiag::choose)
        .def("lane",[](const SplitDiag &d,int g,int i){return d.inverse.at(g).at(i);});

    bindWave(m);
    m.attr("small_playout_search_contract")="terminal-hp-boss-win-turn128-v1";
    m.attr("small_paired_search_contract")="paired-worldlines-eps005-terminal-hp-boss-win-turn128-v1";
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
#else
PYBIND11_MODULE(_full480_cpu,m){
    m.attr("battle_context_bytes")=sizeof(sts::BattleContext);
#ifdef STS_ENABLE_RULE_TRACE
    m.attr("rule_trace")=true;
#else
    m.attr("rule_trace")=false;
#endif
    m.attr("contract")="paired-pcg64-pre-reward-anchor-v1";
    bindWave(m);
}
#endif
