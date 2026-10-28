// Collection-owned upper critic transport. No Torch dependency, game rules,
// approximation or model state: Python supplies the unchanged GPU callback.
#include "neural-observation.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <list>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}

// Walk current registries on every request, including newly replaced modules
// and registered parameters. Holding Python references is not a freeze guard.
void validateFrozen(py::object teacher) {
    std::vector<py::object> pending{teacher};
    std::unordered_set<PyObject *> seen;
    while (!pending.empty()) {
        auto module=std::move(pending.back());pending.pop_back();
        if (!seen.insert(module.ptr()).second) continue;
        if (py::cast<bool>(module.attr("training")))
            throw std::runtime_error("frozen upper was made trainable");
        for (auto item:py::cast<py::dict>(module.attr("_parameters")))
            if (!item.second.is_none() && py::cast<bool>(item.second.attr("requires_grad")))
                throw std::runtime_error("frozen upper was made trainable");
        for (auto item:py::cast<py::dict>(module.attr("_modules")))
            if (!item.second.is_none()) pending.push_back(py::reinterpret_borrow<py::object>(item.second));
    }
}

class UpperValueRuntime {
    struct Field {
        py::str name;
        py::dtype dtype;
        std::vector<py::ssize_t> shape;
        std::size_t bytes;
        int trimmed; // 0=unchanged, 1=deck, 2=relic
    };
    struct Entry { std::string key; float value; std::size_t cost; };
    using Entries = std::list<Entry>;
    std::map<int,std::vector<Field>> schemas_;
    Entries entries_;
    // string_view points into the stable list node. Erase the index before its
    // node; keys are never duplicated and hash collisions use full equality.
    std::unordered_map<std::string_view,Entries::iterator> index_;
    std::map<std::array<int,3>,std::uint64_t> scopes_;
    std::deque<std::array<int,3>> scopeOrder_;
    std::uint64_t nextScope_=0;
    std::size_t limit_, bytes_=0, peak_=0, requests_=0, misses_=0, hits_=0, duplicates_=0, evictions_=0;
    double keySeconds_=0, forwardSeconds_=0, prepareSeconds_=0;
    int combat_, deckCapacity_=0, relicCapacity_=0;
    bool pad_, busy_=false;

    static py::array checked(py::handle value, const Field &field) {
        auto a=py::array::ensure(value,py::array::c_style);
        if (!a || a.ndim()!=static_cast<py::ssize_t>(field.shape.size()) ||
                !std::equal(field.shape.begin(),field.shape.end(),a.shape()))
            throw std::invalid_argument("frozen upper observation shape changed: "+std::string(field.name));
        if (!a.dtype().equal(field.dtype))
            throw std::invalid_argument("frozen upper observation dtype changed: "+std::string(field.name));
        return a;
    }
    static int activeWidth(const py::array &a) {
        const auto *ids=static_cast<const std::int64_t *>(a.data());
        for (int n=static_cast<int>(a.size());n>0;--n) if(ids[n-1]!=0) return n;
        return 1;
    }
    int padded(int width,int capacity,int minimum) const {
        if (!pad_) return width;
        int power=1;while(power<width) power*=2;
        return std::min(capacity,std::max(minimum,power));
    }
    void insert(std::string key,float value,std::size_t cost) {
        if(cost>limit_) return;
        while(!entries_.empty() && bytes_+cost>limit_) {
            const auto &entry=entries_.front();
            index_.erase(std::string_view(entry.key));bytes_-=entry.cost;
            entries_.pop_front();++evictions_;
        }
        entries_.push_back({std::move(key),value,cost});
        auto node=std::prev(entries_.end());index_.emplace(std::string_view(node->key),node);
        bytes_+=cost;peak_=std::max(peak_,bytes_);
    }
public:
    UpperValueRuntime(py::dict schemas,int combat,std::size_t maxBytes,bool pad)
        :limit_(maxBytes),combat_(combat),pad_(pad) {
        for(auto phase:schemas) {
            auto &fields=schemas_[py::cast<int>(phase.first)];
            // Keep checkpoint field order for the GPU's packed input layout.
            // Within this immutable schema, concatenation in this order is
            // still a bijective full-byte key, including every consumed field.
            for(auto item:py::cast<py::dict>(phase.second)) {
                const auto name=py::cast<std::string>(item.first);
                const auto a=py::cast<py::array>(item.second);
                if (a.ndim()<1 || a.size()<1 || (!a.dtype().equal(py::dtype::of<float>()) &&
                        !a.dtype().equal(py::dtype::of<std::int64_t>())))
                    throw std::invalid_argument("unsupported upper input schema: "+name);
                int trim=(name=="deck_ids" || name=="deck_upgrades")?1:
                         (name=="relics" || name=="relic_counters")?2:0;
                if(trim && a.ndim()!=1) throw std::invalid_argument("upper token arrays must be vectors");
                fields.push_back({py::str(name),a.dtype(),{a.shape(),a.shape()+a.ndim()},
                                  static_cast<std::size_t>(a.nbytes()),trim});
                if(name=="deck_ids") deckCapacity_=static_cast<int>(a.size());
                if(name=="relics") relicCapacity_=static_cast<int>(a.size());
            }
        }
        if(!deckCapacity_ || !relicCapacity_) throw std::invalid_argument("upper schema requires deck and relics");
    }
    void clear() {
        if(busy_) throw std::runtime_error("cannot clear pending upper inference");
        index_.clear();entries_.clear();bytes_=0;
        scopes_.clear();scopeOrder_.clear();nextScope_=0;
    }
    py::dict stats() const {
        return py::dict("requests"_a=requests_,"misses"_a=misses_,"hits"_a=hits_,
            "batch_duplicates"_a=duplicates_,"evictions"_a=evictions_,"bytes"_a=bytes_,
            "peak_bytes"_a=peak_,"key_seconds"_a=keySeconds_,"forward_seconds"_a=forwardSeconds_);
    }
    double prepareSeconds() const { return prepareSeconds_; }
    std::size_t maxBytes() const { return limit_; }

    py::array_t<float> forward(py::list rows,py::object teacher,py::function callback) {
        if(busy_) throw std::runtime_error("finish pending upper inference before another submission");
        struct Guard { bool &flag;explicit Guard(bool &f):flag(f){flag=true;}~Guard(){flag=false;} } guard(busy_);
        auto started=Clock::now();validateFrozen(teacher);
        if(rows.empty()) throw std::invalid_argument("upper search requires nonempty rows with scalar phase");
        std::map<int,std::vector<std::size_t>> groups;
        for(std::size_t i=0;i<rows.size();++i) {
            auto phase=py::array::ensure(rows[i]["phase"],py::array::c_style);
            if(!phase || phase.ndim()!=1 || phase.size()!=1 || !phase.dtype().equal(py::dtype::of<std::int64_t>()))
                throw std::invalid_argument("upper search requires nonempty rows with scalar phase");
            int id=static_cast<int>(*static_cast<const std::int64_t *>(phase.data()));
            if(id==combat_ || !schemas_.count(id)) throw std::invalid_argument("upper teacher requires nonempty upper decision rows");
            groups[id].push_back(i);
        }
        py::array_t<float> result(rows.size());auto *out=result.mutable_data();
        prepareSeconds_+=elapsed(started);
        for(const auto &[phase,indices]:groups) {
            started=Clock::now();const auto &fields=schemas_.at(phase);
            std::vector<std::vector<py::array>> arrays(indices.size());
            int deck=1,relic=1;
            for(std::size_t i=0;i<indices.size();++i) {
                auto row=py::cast<py::dict>(rows[indices[i]]);
                for(const auto &field:fields) {
                    auto a=checked(row[field.name],field);
                    if(field.name.equal(py::str("deck_ids"))) deck=std::max(deck,activeWidth(a));
                    if(field.name.equal(py::str("relics"))) relic=std::max(relic,activeWidth(a));
                    arrays[i].push_back(std::move(a));
                }
            }
            const int widths[]={0,padded(deck,deckCapacity_,16),padded(relic,relicCapacity_,4)};
            const int active[]={0,deck,relic};
            std::vector<std::size_t> offsets{0},copied;
            for(const auto &field:fields) {
                offsets.push_back(offsets.back()+(field.trimmed?widths[field.trimmed]*field.dtype.itemsize():field.bytes));
                copied.push_back(field.trimmed?active[field.trimmed]*field.dtype.itemsize():field.bytes);
            }
            prepareSeconds_+=elapsed(started);started=Clock::now();
            const std::array<int,3> signature{phase,widths[1],widths[2]};
            if(!scopes_.count(signature)) {
                if(scopes_.size()>=64) { scopes_.erase(scopeOrder_.front());scopeOrder_.pop_front(); }
                scopes_[signature]=nextScope_++;scopeOrder_.push_back(signature);
            }
            const auto header=scopes_.at(signature);
            const std::size_t prefix=sizeof(header),cost=offsets.back()+sizeof(float)+512;
            std::vector<std::string> keys(indices.size());
            std::vector<std::size_t> missing,source(indices.size());
            std::unordered_map<std::string_view,std::size_t> pending;
            std::vector<bool> found(indices.size(),false);
            for(std::size_t i=0;i<indices.size();++i) {
                auto &key=keys[i];key.assign(prefix+offsets.back(),'\0');std::memcpy(key.data(),&header,prefix);
                for(std::size_t k=0;k<fields.size();++k)
                    std::memcpy(key.data()+prefix+offsets[k],arrays[i][k].data(),copied[k]);
                auto hit=index_.find(key);
                if(limit_ && hit!=index_.end()) {
                    out[indices[i]]=hit->second->value;found[i]=true;++hits_;
                    entries_.splice(entries_.end(),entries_,hit->second);
                } else if(limit_ && pending.count(key)) {
                    source[i]=pending.at(key);++duplicates_;
                } else {
                    source[i]=missing.size();pending.emplace(key,missing.size());missing.push_back(i);
                }
            }
            requests_+=indices.size();misses_+=missing.size();keySeconds_+=elapsed(started);
            if(missing.empty()) continue;
            started=Clock::now();py::dict batch;
            for(std::size_t k=0;k<fields.size();++k) {
                const auto &field=fields[k];std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(missing.size())};
                shape.insert(shape.end(),field.shape.begin(),field.shape.end());
                if(field.trimmed) shape[1]=widths[field.trimmed];
                py::array a(field.dtype,shape);const auto bytes=offsets[k+1]-offsets[k];
                for(std::size_t i=0;i<missing.size();++i)
                    std::memcpy(static_cast<char *>(a.mutable_data())+i*bytes,
                                keys[missing[i]].data()+prefix+offsets[k],bytes);
                batch[field.name]=a;
            }
            auto prediction=py::array_t<float,py::array::c_style>::ensure(callback(batch,py::make_tuple(phase,widths[1],widths[2])));
            if(!prediction || prediction.ndim()!=1 || prediction.size()!=static_cast<py::ssize_t>(missing.size()))
                throw std::invalid_argument("upper critic callback must return one float32 value per missing row");
            forwardSeconds_+=elapsed(started);
            // Own output and cached scalars before another GPU graph replay.
            for(std::size_t i=0;i<indices.size();++i) if(!found[i]) out[indices[i]]=prediction.data()[source[i]];
            for(std::size_t i=0;i<missing.size();++i)
                insert(std::move(keys[missing[i]]),prediction.data()[i],cost);
        }
        return result;
    }
};
}

void bindUpperValue(py::module_ &module) {
    py::class_<UpperValueRuntime>(module,"UpperValueRuntime")
        .def(py::init<py::dict,int,std::size_t,bool>())
        .def("forward",&UpperValueRuntime::forward)
        .def("clear",&UpperValueRuntime::clear)
        .def_property_readonly("stats",&UpperValueRuntime::stats)
        .def_property_readonly("prepare_seconds",&UpperValueRuntime::prepareSeconds)
        .def_property_readonly("max_bytes",&UpperValueRuntime::maxBytes);
}
