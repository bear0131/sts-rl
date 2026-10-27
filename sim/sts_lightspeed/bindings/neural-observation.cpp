#include "neural-observation.h"
#include "compact-observation.h"
#include "sim/search/NeuralSearchBatch.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>

namespace py = pybind11;
namespace {
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

// Vocabulary, printed card data, shapes and scales come from the canonical
// Python codec. Only numeric layout conversion runs here; no game rules.
class NeuralObservationDecoder {
    py::dict shapes_, lut_, stats_, scales_, limits_;
    int offset_, width_;
    bool autoregressive_;
public:
    NeuralObservationDecoder(py::dict shapes, py::dict lut, py::dict stats,
                             py::dict scales, py::dict limits, int offset, int width, bool ar)
        : shapes_(shapes), lut_(lut), stats_(stats), scales_(scales), limits_(limits),
          offset_(offset), width_(width), autoregressive_(ar) {}

    void setCardStats(py::dict stats) { stats_ = std::move(stats); }

    py::dict decode(py::dict packed, py::dict globals, py::object destination) const {
        return decodeRows(packed, globals, destination, nullptr, {}, py::none());
    }

    py::dict decodeNative(sts::search::NeuralSearchBatch &batch, py::dict globals,
                         py::object destination, py::object rows, py::object languageIndex) const {
        auto ownership=batch.acquire();
        std::vector<std::size_t> indices;
        if (rows.is_none()) {
            for (std::size_t i=0;i<batch.requests().size();++i) indices.push_back(i);
        } else {
            for (auto row:py::cast<py::iterable>(rows)) {
                auto index=py::cast<py::ssize_t>(row);
                if(index<0 || static_cast<std::size_t>(index)>=batch.requests().size())
                    throw std::invalid_argument("native observation row out of range");
                indices.push_back(static_cast<std::size_t>(index));
            }
        }
        py::dict packed;
        Array<std::int64_t> owners(indices.size());
        for(std::size_t r=0;r<indices.size();++r) owners.mutable_data()[r]=batch.owners()[indices[r]];
        packed["owners"]=owners;
        return decodeRows(packed, globals, destination, &batch, indices, languageIndex);
    }

    py::dict decodeRows(py::dict packed, py::dict globals, py::object destination,
                       const sts::search::NeuralSearchBatch *native,
                       const std::vector<std::size_t> &indices, py::object languageIndex) const {
        Array<std::int32_t> cat, prefix, legal;
        Array<float> scalar;
        Array<std::int64_t> owners(packed["owners"]);
        const auto n = owners.size();
        auto layout = py::cast<py::dict>(lut_["layout"]);
        const int ps = py::cast<int>(layout["player_status_count"]), ms = py::cast<int>(layout["monster_status_count"]);
        const int cards = py::cast<int>(layout["card_count"]), counters = py::cast<int>(limits_["counters"]);
        const int handStart = 6+ps+counters+1+30+5*ms;
        const int pileStart = handStart+20, selectStart = pileStart+3*cards;
        if (!native) {
            cat=Array<std::int32_t>(packed["categorical"]);prefix=Array<std::int32_t>(packed["selection_prefix"]);
            legal=Array<std::int32_t>(packed["legal_policy_ids"]);scalar=Array<float>(packed["scalars"]);
            if (cat.ndim()!=2) throw std::invalid_argument("native categorical layout mismatch");
            if (cat.shape(1)!=226 || cat.shape(0)!=n || scalar.ndim()!=2 || scalar.shape(0)!=n ||
                scalar.shape(1)!=selectStart+202 || owners.ndim()!=1 ||
                prefix.ndim()!=1 || prefix.size()!=n || legal.ndim()!=2 || legal.shape(0)!=n)
                throw std::invalid_argument("native observation layout mismatch");
        } else if (selectStart+202!=compact_leaf::kFloatSize || counters!=compact_leaf::kCombatCounters ||
                   ps!=compact_leaf::kPlayerStatus || ms!=compact_leaf::kMonsterStatus)
            throw std::invalid_argument("native observation layout mismatch");
        py::dict out;
        std::map<std::string, py::array> arrays;
        for (auto item : shapes_) {
            auto name = py::cast<std::string>(item.first);
            auto prototype = py::cast<py::array>(item.second);
            std::vector<py::ssize_t> shape{n};
            for (int d=0; d<prototype.ndim(); ++d) shape.push_back(prototype.shape(d));
            auto dtype = prototype.dtype();
            py::array array;
            if (destination.is_none()) {
                array = py::array(dtype, shape);
            } else {
                array = py::cast<py::array>(destination[item.first]);
                if (!array.dtype().is(dtype) || !array.writeable() ||
                    !(array.flags() & py::array::c_style) || array.ndim()!=int(shape.size()) || array.shape(0)<n)
                    throw std::invalid_argument("native observation destination layout mismatch");
                for (int d=1; d<array.ndim(); ++d)
                    if (array.shape(d)!=shape[d]) throw std::invalid_argument("native observation destination shape mismatch");
            }
            // Only real rows belong to this request. Padding remains owned by
            // the inference backend, which initializes it from a valid row.
            py::ssize_t rowBytes = array.itemsize();
            for (int d=1; d<array.ndim(); ++d) rowBytes *= array.shape(d);
            // Root globals below overwrite every real row. Clearing their
            // deck/relic/context arrays first only doubles the memory writes.
            if (!globals.contains(item.first))
                std::memset(array.mutable_data(), 0, n * rowBytes);
            arrays.emplace(name, array); out[item.first] = array;
        }
        // Resolve all pointers once per batch, with no dictionary per leaf.
        auto f = [&](const char *key) {
            auto &array=arrays.at(key);
            if (!array.dtype().is(py::dtype::of<float>())) throw std::invalid_argument("native float observation dtype mismatch");
            return static_cast<float *>(array.mutable_data());
        };
        auto i = [&](const char *key) {
            auto &array=arrays.at(key);
            if (!array.dtype().is(py::dtype::of<std::int64_t>())) throw std::invalid_argument("native category observation dtype mismatch");
            return static_cast<std::int64_t *>(array.mutable_data());
        };
        auto *hand=i("hand_ids"), *monster=i("monster_id"), *moves=i("monster_move"),
             *intents=i("monster_intent_cat"), *potions=i("potions"),
             *context=i("select_context"), *selectIds=i("select_card_ids"),
             *selectIndices=i("select_native_indices"), *zone=i("select_zone"), *relics=i("relics"),
             *selectedIds=arrays.count("selected_card_ids") ? i("selected_card_ids") : nullptr;
        auto *player=f("player_scalars"), *pb=f("player_buffs"), *cb=f("combat_counters"),
             *boundary=f("post_enemy_turn_boundary"), *mon=f("monster_scalars"), *mb=f("monster_buffs"),
             *hu=f("hand_upgrades"), *hc=f("hand_costs"), *hd=f("hand_damage"), *hb=f("hand_block"),
             *draw=f("draw_counts"), *discard=f("discard_counts"), *exhaust=f("exhaust_counts"),
             *pick=f("pick_remaining"), *zero=f("can_pick_zero"), *any=f("can_pick_any"),
             *su=f("select_card_upgrades"), *sc=f("select_card_costs"), *selected=f("select_hand_mask"),
             *relicCounters=f("relic_counters"), *gold=f("gold"),
             *selectedUpgrades=arrays.count("selected_card_upgrades") ? f("selected_card_upgrades") : nullptr;
        const auto selectedWidth=selectedIds ? arrays.at("selected_card_ids").shape(1) : 0;
        if ((selectedIds == nullptr) != (selectedUpgrades == nullptr) ||
            (selectedIds && (selectedWidth<2 || arrays.at("selected_card_upgrades").shape(1)!=selectedWidth)))
            throw std::invalid_argument("native selected GRID observation capacity mismatch");
        const auto relicWidth=arrays.at("relics").shape(1);
        const auto pbWidth=arrays.at("player_buffs").shape(1), mbWidth=arrays.at("monster_buffs").shape(2);
        Array<std::int64_t> cardMap(lut_["card"]), monsterMap(lut_["monster"]),
            moveMap(lut_["move"]), potionMap(lut_["potion"]),
            contextMap(lut_["select_context"]), binary(lut_["binary_counters"]),
            liveRelicIds(lut_["live_relic_ids"]), liveColumns(lut_["live_relic_counter_columns"]),
            liveModuli(lut_["live_relic_counter_moduli"]), liveStopTurns(lut_["live_relic_counter_stop_turns"]);
        if (liveRelicIds.size()!=liveColumns.size() || liveRelicIds.size()!=liveModuli.size() ||
            liveRelicIds.size()!=liveStopTurns.size())
            throw std::invalid_argument("native live relic counter word table mismatch");
        auto pc=py::cast<py::tuple>(lut_["player_status_columns"]), mc=py::cast<py::tuple>(lut_["monster_status_columns"]);
        Array<std::int64_t> pn(pc[0]), pt(pc[1]), mn(mc[0]), mt(mc[1]);
        Array<float> damage(stats_["damage"]), block(stats_["block"]);
        auto scale=[&](const char *key) { return py::cast<float>(scales_[key]); };
        const float hpScale=scale("HP"), blockScale=scale("BLOCK"), energyScale=scale("ENERGY"),
            turnScale=scale("TURN"), buffScale=scale("BUFF"), counterScale=scale("COUNTER"),
            dmgScale=scale("DMG"), hitsScale=scale("HITS"), upgradeScale=scale("UPGRADE"), pickScale=scale("PICK"), goldScale=scale("GOLD");
        const int moveLimit=py::cast<int>(limits_["moves"]), intentLimit=py::cast<int>(limits_["intents"]), contextLimit=py::cast<int>(limits_["contexts"]);
        const bool language=native ? !languageIndex.is_none() : packed.contains("language_ids");
        Array<std::int32_t> languageMap;
        Array<std::int32_t> languageIds;
        Array<float> languageFeatures;
        std::int64_t *langIds[2]{};
        float *langCount[2]{}, *langUpgraded[2]{}, *langUpgrades[2]{};
        int slots=0;
        if (language) {
            slots=static_cast<int>(arrays.at("draw_card_ids").shape(1));
            if (native) {
                languageMap=Array<std::int32_t>(languageIndex);
                if(languageMap.ndim()!=1 || languageMap.size()!=compact_leaf::kCardIds || slots!=compact_leaf::kLanguageSlots)
                    throw std::invalid_argument("native language observation layout mismatch");
            } else {
            languageIds=Array<std::int32_t>(packed["language_ids"]); languageFeatures=Array<float>(packed["language_features"]);
            if (languageIds.ndim()!=3 || languageIds.shape(0)!=n || languageIds.shape(1)!=2 || languageIds.shape(2)!=slots ||
                languageFeatures.ndim()!=4 || languageFeatures.shape(0)!=n || languageFeatures.shape(1)!=2 ||
                languageFeatures.shape(2)!=slots || languageFeatures.shape(3)!=3)
                throw std::invalid_argument("native language observation layout mismatch");
            }
            langIds[0]=i("draw_card_ids"); langIds[1]=i("discard_card_ids");
            langCount[0]=f("draw_card_counts"); langCount[1]=f("discard_card_counts");
            langUpgraded[0]=f("draw_card_upgraded_counts"); langUpgraded[1]=f("discard_card_upgraded_counts");
            langUpgrades[0]=f("draw_card_upgrades"); langUpgrades[1]=f("discard_card_upgrades");
        }
        bool *mask=width_ ? static_cast<bool *>(arrays.at("combat_legal").mutable_data()) : nullptr;
        for (auto item : globals) {
            auto source=py::cast<py::array>(item.second); auto &target=arrays.at(py::cast<std::string>(item.first));
            if (!(source.flags() & py::array::c_style) || source.ndim()!=target.ndim() || !source.dtype().is(target.dtype()))
                throw std::invalid_argument("native global observation layout mismatch");
            for (int d=1; d<source.ndim(); ++d)
                if (source.shape(d)!=target.shape(d)) throw std::invalid_argument("native global observation shape mismatch");
            // NumPy permits a zero leading stride on a contiguous singleton
            // batch (value[None, :]). Its row still contains all trailing
            // elements; using strides(0) silently copied zero global bytes.
            py::ssize_t bytes=source.itemsize();
            for (int d=1; d<source.ndim(); ++d) bytes *= source.shape(d);
            for (py::ssize_t r=0; r<n; ++r) {
                const auto owner=owners.data()[r];
                if (owner<0 || owner>=source.shape(0)) throw std::invalid_argument("native observation owner out of range");
                std::memcpy(static_cast<char *>(target.mutable_data())+r*bytes, static_cast<const char *>(source.data())+owner*bytes, bytes);
            }
        }
        auto lookup=[](const Array<std::int64_t> &map, int value,
                       const char *field, bool allowOneAsPad=false) {
            if (value < 0 || value >= map.size()) {
                throw std::invalid_argument(std::string("native ") + field +
                    " enum is outside its exported word table");
            }
            const auto encoded = map.data()[value];
            if (encoded == 0 && value != 0 && !(allowOneAsPad && value == 1)) {
                throw std::invalid_argument(std::string("native ") + field +
                    " identity maps to PAD");
            }
            return encoded;
        };
        // Native rows own an exact candidate count. Their unused slots were
        // zeroed above, so avoid converting 96 padding entries at every leaf.
        // The externally supplied packed decoder still validates all slots.
        const bool sparseSelection = native && cardMap.data()[0] == 0 &&
            upgradeScale > 0 && energyScale > 0 &&
            !globals.contains("select_card_ids") && !globals.contains("select_native_indices") &&
            !globals.contains("select_card_upgrades") && !globals.contains("select_card_costs");
        for (py::ssize_t r=0; r<n; ++r) {
            const auto *request=native ? &native->requests()[indices[r]] : nullptr;
            std::optional<compact_leaf::Row> rawRow;
            if (request) rawRow.emplace(compact_leaf::pack(*request->state));
            const auto *c=request ? rawRow->categorical.data() : cat.data()+r*cat.shape(1);
            const auto *s=request ? rawRow->scalars.data() : scalar.data()+r*scalar.shape(1);
            struct LanguageRow {
                std::array<std::int32_t,2*compact_leaf::kLanguageSlots> ids{};
                std::array<float,6*compact_leaf::kLanguageSlots> features{};
            };
            std::optional<LanguageRow> nativeLanguage;
            if (request && language) {
                nativeLanguage.emplace();
                compact_leaf::packLanguagePile(request->state->cards.drawPile,languageMap.data(),nativeLanguage->ids.data(),nativeLanguage->features.data());
                compact_leaf::packLanguagePile(request->state->cards.discardPile,languageMap.data(),nativeLanguage->ids.data()+slots,nativeLanguage->features.data()+slots*3);
            }
            if (!std::isfinite(s[5]) || s[5]<0 || s[5]>32767 || s[5]!=std::nearbyint(s[5]))
                throw std::invalid_argument("native packed gold is outside the explicit integer domain 0..32767");
            gold[r]=s[5]/goldScale;
            for (int k=0; k<2; ++k)
                if (s[selectStart+198+k]!=0.0f && s[selectStart+198+k]!=1.0f)
                    throw std::invalid_argument("native live relic flag is not boolean");
            const float pocketwatch=s[selectStart+200];
            if (!std::isfinite(pocketwatch) || pocketwatch<0.0f || pocketwatch!=std::nearbyint(pocketwatch))
                throw std::invalid_argument("native Pocketwatch counter is not a nonnegative integer");
            const float neow=s[selectStart+201];
            if (!std::isfinite(neow) || neow<0.0f || neow>3.0f || neow!=std::nearbyint(neow))
                throw std::invalid_argument("native Neow remainder is outside 0..3");
            auto activeRelicWidth = relicWidth;
            while (activeRelicWidth && relics[r*relicWidth+activeRelicWidth-1] == 0)
                --activeRelicWidth;
            for (py::ssize_t k=0; k<liveRelicIds.size(); ++k) {
                const auto column=liveColumns.data()[k];
                if (column < -5 || column >= counters)
                    throw std::invalid_argument("native live relic counter column out of range");
                float value=column == -5 ? neow : column == -4 ? pocketwatch : column == -3 ? s[4]+1.0f : column < 0 ? s[selectStart+197-column] : s[6+ps+column];
                const auto modulo=liveModuli.data()[k], stop=liveStopTurns.data()[k];
                if (modulo) value=std::fmod(value,static_cast<float>(modulo));
                if (stop && value>=stop) value=0.0f;
                if (column!=-1 && column!=-2) value/=counterScale;
                bool owned=false;
                // Keep holes, duplicate IDs and their original slots. Only
                // trailing PADs can be omitted (an ID-zero table still scans
                // the complete row, preserving the decoder's general API).
                const auto scanWidth = liveRelicIds.data()[k] == 0 ? relicWidth : activeRelicWidth;
                for (py::ssize_t slot=0; slot<scanWidth; ++slot)
                    if (relics[r*relicWidth+slot]==liveRelicIds.data()[k]) {
                        relicCounters[r*relicWidth+slot]=value;
                        owned=true;
                    }
                if ((column==-1 || column==-2 || column==-5) && value!=0.0f && !owned)
                    throw std::invalid_argument("native active relic flag has no owned relic");
            }
            for (int k=0; k<10; ++k)
                hand[r*10+k]=lookup(cardMap,c[k],"hand card");
            for (int k=0; k<5; ++k) {
                monster[r*5+k]=lookup(monsterMap,c[10+k],"monster");
                moves[r*5+k]=lookup(moveMap,c[15+k],"monster move");
                if (moves[r*5+k] < 0 || moves[r*5+k] >= moveLimit)
                    throw std::invalid_argument("canonical monster move is outside the observation vocabulary");
                if (c[20+k] < 0 || c[20+k] >= intentLimit)
                    throw std::invalid_argument("native monster intent is outside its word table");
                intents[r*5+k]=c[20+k];
                potions[r*5+k]=lookup(potionMap,c[25+k],"potion",true);
            }
            context[r]=lookup(contextMap,c[30],"selection context");
            if (context[r] < 0 || context[r] >= contextLimit)
                throw std::invalid_argument("canonical selection context is outside the observation vocabulary");
            if (c[223] < 0 || c[223] > 6)
                throw std::invalid_argument("native selection zone is outside its word table");
            zone[r]=c[223];
            for (int k=0; k<2; ++k) {
                const auto selectedId=lookup(cardMap,c[224+k],"selected GRID card");
                if (selectedIds) selectedIds[r*selectedWidth+k]=selectedId;
                else if (selectedId != 0) throw std::invalid_argument(
                    "observation contract cannot represent selected GRID cards");
                const float up=s[selectStart+196+k];
                if (!std::isfinite(up) || up<0 || up!=std::nearbyint(up) || (c[224+k]==0 && up!=0))
                    throw std::invalid_argument("native selected GRID upgrades are invalid");
                if (selectedUpgrades) selectedUpgrades[r*selectedWidth+k]=up/upgradeScale;
                else if (up != 0) throw std::invalid_argument(
                    "observation contract cannot represent selected GRID upgrades");
            }
            const int selectionCount = sparseSelection ? rawRow->selectionCount : 96;
            for (int k=0; k<selectionCount; ++k) {
                selectIds[r*96+k]=lookup(cardMap,c[31+k],"selection card");
                if (c[127+k] < 0 || c[127+k] > 65536)
                    throw std::invalid_argument("native selection handle is outside its declared range");
                selectIndices[r*96+k]=c[127+k];
                su[r*96+k]=s[selectStart+4+k]/upgradeScale; sc[r*96+k]=s[selectStart+100+k]/energyScale;
            }
            auto *p=player+r*6;
            p[0]=std::min(s[0],hpScale)/hpScale; p[1]=std::min(s[1],hpScale)/hpScale;
            p[2]=s[2]/blockScale; p[3]=s[3]/energyScale; p[4]=(s[4]+1.0f)/turnScale; p[5]=s[1] ? s[0]/s[1] : 0.0f;
            for (py::ssize_t k=0; k<pn.size(); ++k) pb[r*pbWidth+pt.data()[k]]=s[6+pn.data()[k]]/buffScale;
            for (int k=0; k<counters; ++k) cb[r*counters+k]=s[6+ps+k]/counterScale;
            for (py::ssize_t k=0; k<binary.size(); ++k) cb[r*counters+binary.data()[k]]=s[6+ps+binary.data()[k]]!=0;
            boundary[r]=s[6+ps+counters];
            if (boundary[r] != 0.0f && boundary[r] != 1.0f)
                throw std::invalid_argument("native post-enemy-turn marker is not boolean");
            for (int k=0; k<5; ++k) {
                const auto *raw=s+6+ps+counters+1+k*6; auto *m=mon+(r*5+k)*6;
                if (c[10+k] == 0 && c[15+k] != 0)
                    throw std::invalid_argument("empty monster slot carries a move");
                if (c[10+k] != 0 && c[15+k] == 0 && raw[0] > 0 && boundary[r] == 0.0f)
                    throw std::invalid_argument("live monster has no move outside a turn boundary");
                m[0]=raw[0]; m[1]=std::min(raw[1],hpScale)/hpScale; m[2]=raw[2] ? raw[1]/raw[2] : 0.0f;
                m[3]=raw[3]/blockScale; m[4]=raw[4]/dmgScale; m[5]=raw[5]/hitsScale;
                for (py::ssize_t j=0; j<mn.size(); ++j)
                    mb[(r*5+k)*mbWidth+mt.data()[j]]=(s[6+ps+counters+1+30+k*ms+mn.data()[j]]/buffScale)*float(raw[0]>0);
            }
            for (int k=0; k<10; ++k) {
                auto pos=r*10+k; const auto upgrade=s[handStart+2*k];
                if (!std::isfinite(upgrade) || upgrade < 0.0f ||
                    upgrade != std::nearbyint(upgrade) || upgrade >= damage.shape(1))
                    throw std::invalid_argument("native hand upgrade is outside the printed-card table");
                hu[pos]=upgrade/upgradeScale; hc[pos]=s[handStart+2*k+1]/energyScale;
                const auto level=static_cast<py::ssize_t>(upgrade);
                const auto slot=hand[pos]*damage.shape(1)+level;
                hd[pos]=damage.data()[slot]/dmgScale; hb[pos]=block.data()[slot]/blockScale;
            }
            std::memcpy(draw+r*cards,s+pileStart,cards*sizeof(float));
            std::memcpy(discard+r*cards,s+pileStart+cards,cards*sizeof(float));
            std::memcpy(exhaust+r*cards,s+pileStart+2*cards,cards*sizeof(float));
            for (int pile=0; pile<(language?2:0); ++pile) for (int k=0; k<slots; ++k) {
                const auto src=((request ? 0 : r)*2+pile)*slots+k, dst=r*slots+k;
                const auto *ids=request ? nativeLanguage->ids.data() : languageIds.data();
                const auto *features=request ? nativeLanguage->features.data() : languageFeatures.data();
                langIds[pile][dst]=ids[src]; langCount[pile][dst]=features[src*3];
                langUpgraded[pile][dst]=features[src*3+1]; langUpgrades[pile][dst]=features[src*3+2]/upgradeScale;
            }
            pick[r]=s[selectStart]/pickScale; zero[r]=s[selectStart+1]; any[r]=s[selectStart+2];
            const int selectionPrefix=request ? request->selectionPrefix : prefix.data()[r];
            if (autoregressive_) {
                if (selectionPrefix < 0 || selectionPrefix >= (1 << 10))
                    throw std::invalid_argument("native selection prefix is outside the hand bitset");
                int count=0;
                for (int k=0; k<10; ++k) { const int bit=(selectionPrefix>>k)&1; selected[r*10+k]=bit; count+=bit; }
                // NumPy's int64 sum promotes this subtraction to float64.
                pick[r]=static_cast<float>(std::max(0.0,double(pick[r])-double(count)/double(pickScale)));
            }
            const auto legalCount=request ? static_cast<py::ssize_t>(request->legalActions.size()) : legal.shape(1);
            if (mask) for (py::ssize_t k=0; k<legalCount; ++k) {
                const auto rawId=request ? sts::search::jointActionId(request->legalActions[k], *request->state)
                    - (request->autoregressivePolicy ? sts::search::kJointEnd : 0) : legal.data()[r*legal.shape(1)+k];
                if (rawId == -1) continue; // explicit absent action slot
                const auto id=rawId-offset_;
                if (id < 0 || id >= width_)
                    throw std::invalid_argument("native policy id outside observation mask");
                mask[r*width_+id]=true;
            }
        }
        return out;
    }
};
} // namespace

void bindNeuralObservation(py::module_ &module) {
    py::class_<NeuralObservationDecoder>(module,"NeuralObservationDecoder")
        .def(py::init<py::dict,py::dict,py::dict,py::dict,py::dict,int,int,bool>())
        .def("set_card_stats", &NeuralObservationDecoder::setCardStats)
        .def("decode_native", &NeuralObservationDecoder::decodeNative,
             py::arg("batch"),py::arg("globals"),py::arg("destination")=py::none(),
             py::arg("rows")=py::none(),py::arg("language_index")=py::none())
        .def("decode",&NeuralObservationDecoder::decode, py::arg("packed"),
             py::arg("globals"), py::arg("destination")=py::none());
}
