#include "combat/BattleContext.h"
#include "sim/search/SmallPlayout.h"
#include "sim/search/PlayoutEntities.h"
#include "game/GameContext.h"
#include "sim/search/Nosl.h"
#include "sim/search/NoslCombatTrainer.h"
#include <random>
#include <cmath>
#include "state_codec.h"
#include "bridge_common.h"
#include "rule_controller.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
static_assert(device_state::TURN_CARD_LIMIT==sts::search::SMALL_MAX_CARDS_PER_TURN);
py::dict search_entity_dict(const std::vector<sts::search::PlayoutEntity>&rows){
 int n=rows.size();py::array_t<int> ids(n),zones(n),monsters(n);py::array_t<float> numbers({n,16});
 for(int i=0;i<n;++i){ids.mutable_data()[i]=rows[i].id;zones.mutable_data()[i]=rows[i].zone;monsters.mutable_data()[i]=rows[i].monster;
  std::copy_n(rows[i].numbers.data(),16,numbers.mutable_data()+i*16);
 }
 py::dict d;d["ids"]=ids;d["zones"]=zones;d["monsters"]=monsters;d["numbers"]=numbers;return d;
}

struct CpuSearchState {
 sts::BattleContext state;
 explicit CpuSearchState(const sts::BattleContext&b):state(b){
#ifdef STS_ENABLE_RULE_TRACE
 state.atomicStepObserver={};
#endif
 }
 explicit CpuSearchState(py::array_t<unsigned long long,py::array::c_style> words){
  device_state::Archive<true> r{words.mutable_data(),int(words.size())};r.field(state);device_state::require(r.at==int(words.size()));
 }
 py::dict snapshot(){
  std::vector<unsigned long long> tmp(8192);
  device_state::Archive<false> w{tmp.data(),int(tmp.size())};w.field(state);
  py::array_t<unsigned long long> words(w.at);std::copy_n(tmp.data(),w.at,words.mutable_data());
  py::dict d;d["words"]=words;d["queue_size"]=state.actionQueue.size;d["max_vector"]=w.max_vector;
  d["portable"]=w.portable&&w.max_vector<=128;
  d["input_state"]=int(state.inputState);d["outcome"]=int(state.outcome);
  return d;
 }
 std::vector<unsigned> actions()const{
  std::vector<unsigned> out;for(auto a:device_state::legal_actions(state))out.push_back(a.bits);return out;
 }
 CpuSearchState variant(int mode)const{
  CpuSearchState x(state);
  if(mode==1||mode==2)x.state.player.setHasRelic<sts::RelicId::RUNIC_DOME>(true);
  if(mode==2){
   for(auto&m:x.state.monsters.arr){
    using Move=std::remove_reference_t<decltype(m.moveHistory[0])>;
    m.moveHistory[0]=static_cast<Move>(0);m.inactiveIntentDamage+=123;m.escapeNext=!m.escapeNext;m.miscInfo+=17;
   }
  }
  if(mode==3){
   x.state.monsters.arr[0]=sts::Monster{};
   x.state.monsters.monstersAlive=0;
   for(auto&m:x.state.monsters.arr)if(m.id!=sts::MonsterId::INVALID&&m.isAlive())++x.state.monsters.monstersAlive;
  }
  return x;
 }

 py::dict observation(int master){
  auto f=sts::search::smallFeatures(state);sts::search::PlayoutEntityObservation obs;
  sts::search::playoutEntities(state,master,f.actions,obs);
  int n=f.actions.size();py::array_t<float> st(32),numbers({n,24});py::array_t<int> keys({n,10}),sources({n,10}),targets(n);py::array_t<unsigned> bits(n);
  std::copy_n(f.state.data(),32,st.mutable_data());
  for(int i=0;i<n;++i){auto&a=f.actions[i];bits.mutable_data()[i]=a.bits;std::copy_n(a.x.data(),24,numbers.mutable_data()+i*24);
   for(int j=0;j<10;++j)keys.mutable_data()[i*10+j]=j<a.keyCount?a.keys[j]:-1;
   std::copy_n(obs.sources[i].data(),10,sources.mutable_data()+i*10);targets.mutable_data()[i]=obs.targets[i];
  }
  py::dict d;d["state"]=st;d["actions"]=numbers;d["keys"]=keys;d["action_bits"]=bits;
  d["entities"]=search_entity_dict(obs.entities);d["source_rows"]=sources;d["target_slots"]=targets;return d;
 }

 py::dict root_plan(unsigned long long nonce,int budget,double temperature,int depth,bool boss){
  if(state.outcome!=sts::Outcome::UNDECIDED||budget<=0||depth<0||depth>200||!std::isfinite(temperature)||temperature<=0)
   throw std::invalid_argument("invalid finite root plan");
  auto legal=device_state::legal_actions(state);if(legal.empty())throw std::logic_error("empty root");
  std::vector<unsigned> bits;std::vector<unsigned long long> semantic;std::vector<int> order;
  for(auto a:legal){bits.push_back(a.bits);semantic.push_back(sts::search::NoslCombatTrainer::strongActionKey(state,a));order.push_back(int(order.size()));}
  std::mt19937_64 shuffle(sts::search::noslMix(nonce^0x5045524dULL));std::shuffle(order.begin(),order.end(),shuffle);
  auto snap=snapshot();py::dict d;d["snapshot"]=snap;d["action_bits"]=bits;d["semantic"]=semantic;d["order"]=order;
  d["nonce"]=nonce;d["budget"]=budget;d["temperature"]=temperature;d["depth_limit"]=depth;d["boss"]=boss;d["max_hp"]=state.player.maxHp;
  d["eligible"]=py::cast<bool>(snap["portable"])&&(state.inputState==sts::InputState::PLAYER_NORMAL||state.inputState==sts::InputState::CARD_SELECT);
  return d;
 }
 CpuSearchState child(unsigned long long seed,unsigned bits,unsigned long long semantic)const{
  CpuSearchState x(state);sts::search::smallRedeterminize(x.state,seed);
  sts::search::Action selected(bits);
  if(!selected.isValidAction(x.state)||sts::search::NoslCombatTrainer::strongActionKey(x.state,selected)!=semantic){
   bool found=false;
   for(auto a:device_state::legal_actions(x.state))if(sts::search::NoslCombatTrainer::strongActionKey(x.state,a)==semantic){selected=a;found=true;break;}
   if(!found)throw std::logic_error("root action semantic mapping failed");
  }
  selected.execute(x.state);return x;
 }
 double terminal_value(bool boss,int maximum)const{
  if(maximum<=0)throw std::invalid_argument("invalid root maximum HP");
  return sts::search::smallTerminalValue(state,boss)/(boss?1.:maximum);
 }

 void step(unsigned bits){
  auto a=device_state::legal_actions(state);
  device_state::require(std::any_of(a.begin(),a.end(),[&](auto x){return x.bits==bits;}));
  sts::search::Action(bits).execute(state);
 }
};
PYBIND11_MODULE(gpu_search_bridge,m){
 m.def("inventory",[](const sts::GameContext&gc){py::dict d;d["inventory"]=search_entity_dict(sts::search::playoutInventory(gc));d["master_deck_size"]=gc.deck.size();return d;});
 m.attr("state_schema")=2;
 m.attr("cpu_battle_bytes")=sizeof(sts::BattleContext);
 m.attr("arity")=py::make_tuple(device_state::arity<sts::BattleContext>(),device_state::arity<sts::Player>(),
    device_state::arity<sts::Monster>(),device_state::arity<sts::CardManager>(),device_state::arity<sts::CardSelectInfo>());
 py::class_<CpuSearchState>(m,"State")
 .def(py::init<const sts::BattleContext&>())
 .def(py::init<py::array_t<unsigned long long,py::array::c_style>>())
 .def("redeterminize",[](CpuSearchState&x,unsigned long long seed){sts::search::smallRedeterminize(x.state,seed);})
 .def("rule_loop",[](CpuSearchState&x,unsigned long long seed,int depth){auto s=device_state::rule_loop(x.state,seed,depth);return std::vector<unsigned>{s.actions,s.decisions,s.forced,s.outcome,s.depth_cut,s.active_pending,s.max_actions,0};})
 .def("root_plan",&CpuSearchState::root_plan)
 .def("child",&CpuSearchState::child)
 .def("terminal_value",&CpuSearchState::terminal_value)
 .def("clone",[](const CpuSearchState&x){return CpuSearchState(x.state);})
 .def("variant",&CpuSearchState::variant)
 .def("observation",&CpuSearchState::observation)
 .def("snapshot",&CpuSearchState::snapshot).def("actions",&CpuSearchState::actions).def("step",&CpuSearchState::step);
}
