// Numeric adapter for deck_context.project. No simulator transitions or cache.
#pragma once
#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace sts::deck_observation {
namespace py = pybind11;

// Collection-owned array bindings. Layout is fixed for the scope; values may
// change between calls. Keep the arrays alive and reject resized storage before
// any output is written, rather than rebuilding Python field metadata per row.
struct BoundObservationArray {
    py::array owner;
    const void *address;
    std::vector<py::ssize_t> shape;
    py::dtype dtype;
    explicit BoundObservationArray(py::array value)
        : owner(value), address(value.data()),
          shape(value.shape(), value.shape()+value.ndim()), dtype(value.dtype()) {
        if (!(value.flags() & py::array::c_style) || value.ndim()<1
            || !std::strchr("biufc",value.dtype().kind()))
            throw std::invalid_argument("observation binding requires contiguous numeric arrays");
    }
    void validate() const {
        if (owner.data()!=address || owner.ndim()!=py::ssize_t(shape.size())
            || !owner.dtype().equal(dtype) || !(owner.flags() & py::array::c_style)
            || !std::equal(shape.begin(),shape.end(),owner.shape()))
            throw std::invalid_argument("bound observation storage changed");
    }
};

class ObservationRowCopy {
    struct Field {
        BoundObservationArray destination, source;
        size_t elements, source_bytes, destination_bytes;
        bool integer_to_float, bool_to_float;
    };
    std::vector<Field> fields_;
    int64_t source_count_=-1, destination_count_=-1;
public:
    ObservationRowCopy(py::dict destination, py::dict source, int axes) {
        if (axes<1 || axes>2 || destination.empty() || destination.size()!=source.size())
            throw std::invalid_argument("invalid observation row copy geometry");
        for (auto item:destination) {
            if (!source.contains(item.first)) throw std::invalid_argument("observation copy schema mismatch");
            auto dst=py::cast<py::array>(item.second),src=py::cast<py::array>(source[item.first]);
            const bool itf=dst.dtype().equal(py::dtype::of<float>()) && src.dtype().equal(py::dtype::of<int64_t>());
            const bool btf=dst.dtype().equal(py::dtype::of<float>()) && src.dtype().equal(py::dtype::of<bool>());
            if (!dst.writeable() || src.ndim()<1 || dst.ndim()!=src.ndim()+axes-1
                || (!dst.dtype().equal(src.dtype()) && !itf && !btf))
                throw std::invalid_argument("observation copy requires compatible contiguous arrays");
            size_t elements=1;
            for (int dim=1;dim<src.ndim();++dim) {
                if (src.shape(dim)!=dst.shape(dim+axes-1)) throw std::invalid_argument("observation copy field shape mismatch");
                elements*=src.shape(dim);
            }
            int64_t count=dst.shape(0)*(axes==2 ? dst.shape(1) : 1);
            if (source_count_<0) { source_count_=src.shape(0);destination_count_=count; }
            if (source_count_!=src.shape(0) || destination_count_!=count)
                throw std::invalid_argument("observation field row counts differ");
            fields_.push_back({BoundObservationArray(dst),BoundObservationArray(src),elements,
                elements*src.itemsize(),elements*dst.itemsize(),itf,btf});
        }
    }
    void copy(py::array_t<int64_t,py::array::c_style> destination_rows,
              py::array_t<int64_t,py::array::c_style> source_rows) const {
        if (destination_rows.ndim()!=1 || source_rows.ndim()!=1 || source_rows.size()!=destination_rows.size())
            throw std::invalid_argument("invalid observation row copy geometry");
        for (const auto &f:fields_) {
            f.source.validate();f.destination.validate();
            if (!f.destination.owner.writeable()) throw std::invalid_argument("observation destination became readonly");
        }
        for (py::ssize_t i=0;i<source_rows.size();++i)
            if (source_rows.data()[i]<0 || source_rows.data()[i]>=source_count_
                || destination_rows.data()[i]<0 || destination_rows.data()[i]>=destination_count_)
                throw std::invalid_argument("observation row copy index out of range");
        py::gil_scoped_release release;
        for (const auto &f:fields_) for (py::ssize_t row=0;row<source_rows.size();++row) {
            auto *out=static_cast<char*>(const_cast<void*>(f.destination.address))+destination_rows.data()[row]*f.destination_bytes;
            const auto *in=static_cast<const char*>(f.source.address)+source_rows.data()[row]*f.source_bytes;
            if (f.integer_to_float) {
                auto *a=reinterpret_cast<float*>(out);auto *b=reinterpret_cast<const int64_t*>(in);
                for (size_t i=0;i<f.elements;++i)a[i]=static_cast<float>(b[i]);
            } else if (f.bool_to_float) {
                auto *a=reinterpret_cast<float*>(out);auto *b=reinterpret_cast<const bool*>(in);
                for (size_t i=0;i<f.elements;++i)a[i]=static_cast<float>(b[i]);
            } else std::memcpy(out,in,f.source_bytes);
        }
    }
};

class ObservationColumns {
    struct Field { BoundObservationArray source; size_t offset,bytes; };
    struct Group { py::dtype dtype;size_t width=0;std::vector<Field> fields; };
    std::vector<Group> groups_;
    int64_t count_=-1;
public:
    ObservationColumns(py::dict source, const std::vector<std::vector<std::string>> &groups) {
        size_t fields=0;
        std::unordered_set<std::string> seen;
        for (const auto &names:groups) {
            if (names.empty()) throw std::invalid_argument("empty observation column group");
            Group group{py::cast<py::array>(source[py::str(names[0])]).dtype()};
            for (const auto &name:names) {
                if (!seen.insert(name).second) throw std::invalid_argument("duplicate observation column field");
                auto array=py::cast<py::array>(source[py::str(name)]);
                BoundObservationArray binding(array);
                if (!array.dtype().equal(group.dtype)) throw std::invalid_argument("observation column dtype mismatch");
                if (count_<0)count_=array.shape(0);
                if (count_!=array.shape(0)) throw std::invalid_argument("observation field row counts differ");
                size_t width=1;
                for (int i=1;i<array.ndim();++i)width*=array.shape(i);
                group.fields.push_back({std::move(binding),group.width*array.itemsize(),width*array.itemsize()});
                group.width+=width;++fields;
            }
            groups_.push_back(std::move(group));
        }
        if (fields!=source.size())throw std::invalid_argument("observation column schema mismatch");
    }
    py::list pack(py::array_t<int64_t,py::array::c_style> rows) const {
        if (rows.ndim()!=1)throw std::invalid_argument("observation column rows must be a vector");
        for (const auto &g:groups_)for (const auto &f:g.fields)f.source.validate();
        for (py::ssize_t i=0;i<rows.size();++i)
            if (rows.data()[i]<0 || rows.data()[i]>=count_)throw std::invalid_argument("observation column index out of range");
        py::list result;
        for (const auto &g:groups_) {
            py::array out(g.dtype,std::vector<py::ssize_t>{rows.size(),py::ssize_t(g.width)});
            auto *dst=static_cast<char*>(out.mutable_data());size_t stride=g.width*out.itemsize();
            {
                py::gil_scoped_release release;
                for (const auto &f:g.fields) for (py::ssize_t i=0;i<rows.size();++i)
                    std::memcpy(dst+i*stride+f.offset,static_cast<const char*>(f.source.address)+rows.data()[i]*f.bytes,f.bytes);
            }
            result.append(out);
        }
        return result;
    }
};

inline std::vector<int64_t> filteredActionTies(
    py::array_t<double,py::array::c_style|py::array::forcecast> logits,
    py::array_t<double,py::array::c_style|py::array::forcecast> values,
    py::array_t<double,py::array::c_style|py::array::forcecast> visits) {
    if (logits.ndim()!=1 || values.ndim()!=1 || visits.ndim()!=1
        || logits.size()!=values.size() || values.size()!=visits.size())
        throw std::invalid_argument("search choice requires finite aligned scores and visited actions");
    double best=-std::numeric_limits<double>::infinity(),rank=best;
    bool visited=false;
    for (py::ssize_t i=0;i<logits.size();++i) {
        if (!std::isfinite(logits.data()[i]) || !std::isfinite(values.data()[i]))
            throw std::invalid_argument("search choice requires finite aligned scores and visited actions");
        if (visits.data()[i]>0) { visited=true;best=std::max(best,values.data()[i]); }
    }
    if (!visited)throw std::invalid_argument("search choice requires finite aligned scores and visited actions");
    std::vector<int64_t> tied;
    for (py::ssize_t i=0;i<logits.size();++i) if (visits.data()[i]>0 && values.data()[i]>=best-.005) {
        if (logits.data()[i]>rank) { rank=logits.data()[i];tied.clear(); }
        if (logits.data()[i]==rank)tied.push_back(i);
    }
    return tied;
}

// Exact byte copies for complete PPO observations. The source and destination
// keep their public dictionary schema; only the per-field Python loop moves.
inline void copyObservationRows(py::dict destination, py::dict source,
    py::array_t<int64_t, py::array::c_style> destination_rows,
    py::array_t<int64_t, py::array::c_style> source_rows, int destination_axes) {
    ObservationRowCopy(std::move(destination),std::move(source),destination_axes)
        .copy(std::move(destination_rows),std::move(source_rows));
}

class DeckContextEncoder {
    std::vector<int> cards_;
    int vocab_, slots_;
    double upgrade_scale_, misc_scale_;
public:
    DeckContextEncoder(std::vector<int> cards, int vocab, int slots,
                       double upgrade_scale, double misc_scale)
        : cards_(std::move(cards)), vocab_(vocab), slots_(slots),
          upgrade_scale_(upgrade_scale), misc_scale_(misc_scale) {
        if (vocab <= 0 || slots < Deck::MAX_SIZE || !(upgrade_scale > 0) || !(misc_scale > 0)
            || !std::isfinite(upgrade_scale) || !std::isfinite(misc_scale))
            throw std::invalid_argument("invalid deck projection geometry");
        for (int card: cards_) if (card < -1 || card >= vocab_)
            throw std::invalid_argument("deck projection vocabulary index out of range");
    }
    py::dict encode(const std::vector<const GameContext*>& games,
                    const std::vector<const BattleContext*>& battles) const {
        if (games.size() != battles.size()) throw std::invalid_argument("deck projection batch mismatch");
        const py::ssize_t n = games.size();
        py::array_t<float> counts({n, py::ssize_t(3), py::ssize_t(vocab_)});
        py::array_t<float> upgrades({n, py::ssize_t(3), py::ssize_t(vocab_)});
        py::array_t<float> misc({n, py::ssize_t(slots_)});
        float *cp = counts.mutable_data(), *up = upgrades.mutable_data(), *mp = misc.mutable_data();
        std::fill_n(cp, counts.size(), 0.f); std::fill_n(up, upgrades.size(), 0.f);
        std::fill_n(mp, misc.size(), 0.f);
        {
            py::gil_scoped_release release;
            for (py::ssize_t row = 0; row < n; ++row) {
                if (!games[row] || !battles[row]) throw std::invalid_argument("missing live deck projection root");
                const auto &gc = *games[row]; const auto &bc = *battles[row];
                const int deck_size = gc.deck.size();
                if (deck_size > slots_) throw std::invalid_argument("master deck exceeds observation capacity");
                float *m = mp + row * slots_;
                for (int i = 0; i < deck_size; ++i) m[i] = float(gc.deck.cards[i].misc / misc_scale_);
                // Same physical master-slot overlay and traversal order as
                // permanent_card_misc; temporary upgrades are never persisted.
                const auto overlay = [&](const CardInstance &card) {
                    if (card.getId() == CardId::RITUAL_DAGGER && card.uniqueId >= 0
                        && card.uniqueId < deck_size)
                        m[card.uniqueId] = float(card.specialData / misc_scale_);
                };
                int zone = 0;
                for (const auto *pile : {&bc.cards.drawPile, &bc.cards.discardPile, &bc.cards.exhaustPile}) {
                    float *c = cp + (row * 3 + zone) * vocab_;
                    float *u = up + (row * 3 + zone) * vocab_;
                    for (const auto &card: *pile) {
                        const int id = int(card.getId());
                        if (id < 0 || id >= int(cards_.size())) throw std::invalid_argument("unmapped native card");
                        const int index = cards_[id], count = card.getUpgradeCount();
                        if (index < 0) throw std::invalid_argument("invalid card in a public pile");
                        c[index] += count > 0;
                        // Python reference adds a double then stores float32
                        // after each card, not once after aggregating a pile.
                        u[index] = float(double(u[index]) + double(count) / upgrade_scale_);
                        overlay(card);
                    }
                    ++zone;
                }
                for (int i = 0; i < bc.cards.cardsInHand; ++i) overlay(bc.cards.hand[i]);
            }
        }
        py::dict result; result["pile_upgraded_counts"] = counts;
        result["pile_upgrade_sums"] = upgrades; result["outer_deck_misc"] = misc;
        return result;
    }
};
inline void bindDeckContextEncoder(py::module_ &m) {
    m.def("copy_observation_rows", &copyObservationRows);
    py::class_<ObservationRowCopy>(m,"ObservationRowCopy")
        .def(py::init<py::dict,py::dict,int>()).def("copy",&ObservationRowCopy::copy);
    py::class_<ObservationColumns>(m,"ObservationColumns")
        .def(py::init<py::dict,const std::vector<std::vector<std::string>>&>())
        .def("pack",&ObservationColumns::pack);
    m.def("filtered_action_ties",&filteredActionTies);
    py::class_<DeckContextEncoder>(m, "DeckContextEncoder")
        .def(py::init<std::vector<int>, int, int, double, double>())
        .def("encode", &DeckContextEncoder::encode);
}
}
