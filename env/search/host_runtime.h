// CPU-only hot paths of the resident search owner. No model or game rules.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <list>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace resident_host {
namespace py = pybind11;
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

struct Handle { int slot; int64_t generation; py::object key; };
class RootLedger {
    std::priority_queue<int,std::vector<int>,std::greater<int>> free_;
    std::vector<int64_t> generations_;
    std::vector<double> deadlines_;
public:
    int capacity;
    py::dict pending;
    explicit RootLedger(int n): generations_(n>0?n:0),deadlines_(n>0?n:0,std::numeric_limits<double>::infinity()),capacity(n) {
        if(n<1)throw py::value_error("positive root capacity required");
        for(int i=0;i<n;++i)free_.push(i);
    }
    std::vector<Handle> claim(py::iterable input) {
        py::list keys(input);py::set unique(keys);
        if(py::len(unique)!=py::len(keys))throw py::value_error("root key already in flight");
        for(auto k:keys)if(pending.contains(k))throw py::value_error("root key already in flight");
        if(py::len(keys)>free_.size())throw std::overflow_error("no free root slots");
        std::vector<Handle> result;result.reserve(py::len(keys));
        for(auto k:keys){int slot=free_.top();free_.pop();Handle h{slot,++generations_[slot],py::reinterpret_borrow<py::object>(k)};
            pending[k]=py::cast(h);result.push_back(std::move(h));}
        return result;
    }
    void release(const Handle& h) {
        if(!pending.contains(h.key))throw py::value_error("stale or unknown root handle");
        auto current=pending[h.key].cast<Handle>();
        if(current.slot!=h.slot||current.generation!=h.generation)throw py::value_error("stale or unknown root handle");
        pending.attr("pop")(h.key);deadlines_[h.slot]=std::numeric_limits<double>::infinity();free_.push(h.slot);
    }
    void set_deadlines(const std::vector<Handle>& handles,Array<double> times) {
        if(handles.size()!=size_t(times.size()))throw py::value_error("deadline count mismatch");
        for(const auto& h:handles)if(!pending.contains(h.key)||pending[h.key].cast<Handle>().generation!=h.generation)
            throw py::value_error("stale root deadline");
        for(size_t i=0;i<handles.size();++i)deadlines_[handles[i].slot]=times.data()[i];
    }
    double deadline()const {return *std::min_element(deadlines_.begin(),deadlines_.end());}
    std::vector<Handle> completed(Array<int32_t> left)const {
        if(left.size()!=capacity)throw py::value_error("root completion layout mismatch");
        std::vector<Handle> result;for(auto item:pending){auto h=item.second.cast<Handle>();if(left.data()[h.slot]==0)result.push_back(std::move(h));}return result;
    }
};

struct Lease {py::object key;int64_t serial;int count;std::vector<int> pages;};
class DevicePages {
    int roots_,budget_,bytes_,page_,growth_,capacity_=0,max_pages_,used_jobs_=0,used_pages_=0,peak_jobs_=0,peak_pages_=0;
    int64_t serial_=0;
    bool closed_=false;
    py::function allocator_;
    py::list blocks_;
    py::dict owned_;
    std::vector<uint64_t> addresses_;
    std::priority_queue<int,std::vector<int>,std::greater<int>> free_;
public:
    DevicePages(int roots,int budget,int bytes,py::function allocator,int page=16,int growth=256):
      roots_(roots),budget_(budget),bytes_(bytes),page_(page),growth_(growth),
      max_pages_(0),allocator_(allocator) {
        if(std::min({roots,budget,bytes,page,growth})<1)throw py::value_error("positive arena limits required");
        max_pages_=roots*((budget+page-1)/page);
    }
    py::tuple acquire(py::iterable input) {
        if(closed_)throw std::runtime_error("arena closed");
        py::list specs(input);py::set keys;int required=0;
        for(auto spec:specs){auto row=py::cast<py::tuple>(spec);auto key=row[0];int n=row[1].cast<int>();
            if(keys.contains(key)||owned_.contains(key))throw py::value_error("page owner already exists");
            keys.add(key);if(n<1||n>budget_)throw py::value_error("invalid per-root budget");required+=(n+page_-1)/page_;}
        if(py::len(owned_)+py::len(specs)>size_t(roots_))throw py::value_error("root count exceeds arena contract");
        int needed=std::max(0,required-int(free_.size()));
        if(needed>max_pages_-capacity_)throw std::overflow_error("logical budgets exceed arena");
        if(needed){int count=std::min(max_pages_-capacity_,((needed+growth_-1)/growth_)*growth_);
            py::object block=allocator_(count*page_);uint64_t base=block.attr("data_ptr")().cast<uint64_t>();
            blocks_.append(block);addresses_.reserve(capacity_+count);
            for(int i=0;i<count;++i){addresses_.push_back(base+uint64_t(i)*page_*bytes_);free_.push(capacity_+i);}capacity_+=count;}
        py::dict leases,pointers;
        for(auto spec:specs){auto row=py::cast<py::tuple>(spec);auto key=py::reinterpret_borrow<py::object>(row[0]);int n=row[1].cast<int>();
            Lease lease{key,++serial_,n,{}};int count=(n+page_-1)/page_;lease.pages.reserve(count);
            for(int i=0;i<count;++i){lease.pages.push_back(free_.top());free_.pop();}
            Array<uint64_t> pointer(n);auto out=pointer.mutable_data();
            {py::gil_scoped_release unlocked;for(int i=0;i<n;++i)out[i]=addresses_[lease.pages[i/page_]]+uint64_t(i%page_)*bytes_;}
            owned_[key]=py::cast(lease);leases[key]=py::cast(lease);pointers[key]=std::move(pointer);
            used_jobs_+=n;used_pages_+=count;}
        peak_jobs_=std::max(peak_jobs_,used_jobs_);peak_pages_=std::max(peak_pages_,used_pages_);
        return py::make_tuple(leases,pointers);
    }
    void release(py::iterable input) {
        if(closed_)throw std::runtime_error("arena closed");
        auto leases=py::cast<std::vector<Lease>>(py::list(input));py::set keys;
        for(const auto& l:leases){if(keys.contains(l.key)||!owned_.contains(l.key)||owned_[l.key].cast<Lease>().serial!=l.serial)
                throw py::value_error("stale, duplicate, or unknown page lease");keys.add(l.key);}
        for(const auto& l:leases){owned_.attr("pop")(l.key);used_jobs_-=l.count;used_pages_-=int(l.pages.size());for(int p:l.pages)free_.push(p);}
    }
    py::dict statistics()const {
        py::dict d;d["page_size"]=page_;d["capacity_pages"]=capacity_;d["max_pages"]=max_pages_;
        d["free_pages"]=free_.size();d["owners"]=py::len(owned_);d["leased_jobs"]=used_jobs_;d["leased_pages"]=used_pages_;
        d["peak_leased_jobs"]=peak_jobs_;d["peak_leased_pages"]=peak_pages_;d["blocks"]=py::len(blocks_);d["state_bytes"]=bytes_;
        d["allocated_state_bytes"]=closed_?int64_t(0):int64_t(capacity_)*page_*bytes_;
        d["peak_allocated_state_bytes"]=int64_t(capacity_)*page_*bytes_;d["closed"]=closed_;return d;
    }
    void close(){if(closed_)return;if(py::len(owned_))throw std::runtime_error("cannot release live battle-state blocks");
        blocks_.attr("clear")();addresses_.clear();closed_=true;}
};

// The anchor consumer has exactly one FP32 output. Full typed input bytes and
// namespaces form the key; neither hashes alone nor state IDs establish a hit.
class ExactValueCache {
    struct Entry{float value;size_t cost;std::list<std::string>::iterator order;};
    size_t limit_,bytes_=0,peak_=0;
    uint64_t requests_=0,misses_=0,hits_=0,duplicates_=0,evictions_=0,next_scope_=0;
    double key_seconds_=0,forward_seconds_=0;
    py::dict scopes_;
    std::list<std::string> order_;
    std::unordered_map<std::string,Entry> entries_;
public:
    explicit ExactValueCache(int64_t limit):limit_(limit){if(limit<0)throw py::value_error("cache capacity must be nonnegative");}
    void clear(){entries_.clear();order_.clear();scopes_.clear();next_scope_=0;bytes_=0;}
    py::dict statistics()const {
        py::dict d;d["requests"]=requests_;d["misses"]=misses_;d["hits"]=hits_;d["batch_duplicates"]=duplicates_;
        d["evictions"]=evictions_;d["bytes"]=bytes_;d["peak_bytes"]=peak_;d["key_seconds"]=key_seconds_;d["forward_seconds"]=forward_seconds_;return d;
    }
    py::tuple forward(py::dict observations,py::function callback,py::object name_space,bool indexed) {
        auto started=Clock::now();py::list names=py::module_::import("builtins").attr("sorted")(observations);
        if(!py::len(names))throw py::value_error("cache inputs require a nonempty batch");
        std::vector<py::array> arrays;std::vector<size_t> widths;py::list schema;
        ssize_t count=-1;size_t width=sizeof(uint64_t);
        for(auto name:names){auto a=py::array::ensure(observations[name],py::array::c_style);
            if(!a||a.ndim()<1||(count>=0&&a.shape(0)!=count))throw py::value_error("cache batch mismatch");
            count=a.shape(0);if(count<1)throw py::value_error("cache inputs require a nonempty batch");
            py::tuple shape(a.ndim()-1);for(int i=1;i<a.ndim();++i)shape[i-1]=a.shape(i);
            schema.append(py::make_tuple(name,shape,a.dtype().attr("str")));
            size_t row=a.nbytes()/count;arrays.push_back(a);widths.push_back(row);width+=row;}
        if(!limit_){requests_+=count;misses_+=count;Array<int64_t> take(count);for(ssize_t i=0;i<count;++i)take.mutable_data()[i]=i;auto result=(indexed?callback(observations,take):callback(observations)).cast<py::tuple>();forward_seconds_+=elapsed(started);return result;}
        auto scope_key=py::make_tuple(name_space,py::tuple(schema));
        if(!scopes_.contains(scope_key)){if(py::len(scopes_)>=64){py::object first=py::reinterpret_borrow<py::object>(scopes_.begin()->first);scopes_.attr("pop")(first);}scopes_[scope_key]=next_scope_++;}
        uint64_t scope=scopes_[scope_key].cast<uint64_t>();
        Array<float> output(count);auto out=output.mutable_data();
        std::vector<std::string> keys(count);std::vector<int64_t> missing,which(count,-1);
        std::unordered_map<std::string,int64_t> pending;
        {py::gil_scoped_release unlocked;
            for(ssize_t i=0;i<count;++i){auto& key=keys[i];key.resize(width);std::memcpy(key.data(),&scope,sizeof(scope));size_t offset=sizeof(scope);
                for(size_t j=0;j<arrays.size();++j){std::memcpy(key.data()+offset,static_cast<const char*>(arrays[j].data())+i*widths[j],widths[j]);offset+=widths[j];}
                auto found=entries_.find(key);
                if(found!=entries_.end()){out[i]=found->second.value;order_.splice(order_.end(),order_,found->second.order);++hits_;}
                else {auto [it,inserted]=pending.emplace(key,missing.size());which[i]=it->second;
                    if(inserted)missing.push_back(i);else ++duplicates_;}}
            requests_+=count;misses_+=missing.size();}
        key_seconds_+=elapsed(started);
        if(!missing.empty()){
            started=Clock::now();Array<int64_t> take(missing.size());std::copy(missing.begin(),missing.end(),take.mutable_data());
            py::dict selected;for(size_t j=0;j<arrays.size();++j)selected[names[j]]=arrays[j].attr("__getitem__")(take);
            auto result=(indexed?callback(selected,take):callback(selected)).cast<py::tuple>();auto logits=py::array::ensure(result[0]);Array<float> values(result[1]);
            if(logits.ndim()!=2||logits.shape(1)!=0||values.size()!=ssize_t(missing.size()))throw py::value_error("anchor cache requires one scalar per row");
            forward_seconds_+=elapsed(started);auto ptr=values.data();
            {py::gil_scoped_release unlocked;
                for(size_t j=0;j<missing.size();++j){auto& key=keys[missing[j]];size_t cost=key.size()-sizeof(scope)+sizeof(float)+512;
                    if(cost<=limit_){while(bytes_+cost>limit_&&!order_.empty()){auto found=entries_.find(order_.front());bytes_-=found->second.cost;entries_.erase(found);order_.pop_front();++evictions_;}
                        order_.push_back(key);entries_.emplace(key,Entry{ptr[j],cost,std::prev(order_.end())});bytes_+=cost;peak_=std::max(peak_,bytes_);}}
                for(ssize_t i=0;i<count;++i)if(which[i]>=0)out[i]=ptr[which[i]];}
        }
        return py::make_tuple(Array<float>(std::vector<ssize_t>{count,0}),output);
    }
};

inline py::tuple pack_arrays(py::list input) {
    std::vector<py::array> arrays;std::vector<int64_t> offsets;std::vector<size_t> bytes;size_t size=0;
    for(auto a:input){auto array=py::array::ensure(a,py::array::c_style);if(!array)throw py::value_error("numeric array required");
        size=(size+7)/8*8;offsets.push_back(size);bytes.push_back(array.nbytes());size+=bytes.back();arrays.push_back(array);}
    Array<uint8_t> packed(size);auto out=packed.mutable_data();
    {py::gil_scoped_release unlocked;for(size_t i=0;i<arrays.size();++i)std::memcpy(out+offsets[i],arrays[i].data(),bytes[i]);}
    return py::make_tuple(packed,offsets);
}

inline py::tuple pack_snapshots(py::list snapshots,Array<int64_t> slots,int capacity) {
    if(slots.size()!=ssize_t(py::len(snapshots)))throw py::value_error("invalid root slot lease");
    std::vector<Array<uint64_t>> rows;ssize_t width=0;std::unordered_set<int64_t> unique;
    for(ssize_t i=0;i<slots.size();++i){auto slot=slots.data()[i];
        if(slot<0||slot>=capacity||!unique.insert(slot).second)throw py::value_error("invalid root slot lease");
        auto state=snapshots[i].cast<py::dict>();
        if(!state["portable"].cast<bool>())throw py::value_error("pending callback root must use native fallback");
        Array<uint64_t> row(state["words"]);width=std::max(width,row.size());rows.push_back(row);}
    Array<uint64_t> words({slots.size(),width});Array<int32_t> lengths(slots.size());
    auto out=words.mutable_data();auto sizes=lengths.mutable_data();
    {py::gil_scoped_release unlocked;std::fill(out,out+words.size(),uint64_t(0));
        for(size_t i=0;i<rows.size();++i){sizes[i]=rows[i].size();std::memcpy(out+i*width,rows[i].data(),rows[i].size()*sizeof(uint64_t));}}
    return py::make_tuple(words,lengths);
}

inline py::array stack_fields(py::list observations,py::list names) {
    if(!py::len(observations)||!py::len(names))throw py::value_error("nonempty observation fields required");
    std::vector<py::array> arrays;std::vector<ssize_t> sizes;std::vector<std::vector<ssize_t>> shapes;
    py::dtype dtype=py::array::ensure(observations[0].cast<py::dict>()[names[0]]).dtype();
    ssize_t width=0;
    for(auto name:names){auto a=py::array::ensure(observations[0].cast<py::dict>()[name],py::array::c_style);
        sizes.push_back(a.size());shapes.emplace_back(a.shape(),a.shape()+a.ndim());width+=a.size();}
    for(auto observation:observations){auto row=py::cast<py::dict>(observation);size_t j=0;
        for(auto name:names){auto a=py::array::ensure(row[name],py::array::c_style);
            if(!a||a.size()!=sizes[j]||!a.dtype().equal(dtype)||std::vector<ssize_t>(a.shape(),a.shape()+a.ndim())!=shapes[j])
                throw py::value_error("observation field layout changed");++j;arrays.push_back(a);}}
    py::array result(dtype,std::vector<ssize_t>{ssize_t(py::len(observations)),width});auto out=static_cast<char*>(result.mutable_data());
    size_t itemsize=dtype.itemsize();
    {py::gil_scoped_release unlocked;for(const auto& a:arrays){size_t bytes=a.size()*itemsize;std::memcpy(out,a.data(),bytes);out+=bytes;}}
    return result;
}

// Exact row interning shared by static context identity and projected anchors.
// IDs never repeat; eviction may only add computation, never create an alias.
class ByteIntern {
    struct Entry{int64_t id;size_t cost;std::list<std::string>::iterator order;};
    size_t limit_,bytes_=0;int64_t next_=0;
    std::list<std::string> order_;std::unordered_map<std::string,Entry> entries_;
public:
    explicit ByteIntern(size_t limit):limit_(limit){}
    Array<int64_t> intern(py::list input){
        std::vector<py::array> arrays;std::vector<size_t> widths;ssize_t count=-1;size_t width=0;std::string schema;
        for(auto item:input){auto a=py::array::ensure(item,py::array::c_style);
            if(!a||a.ndim()<1||(count>=0&&count!=a.shape(0)))throw py::value_error("intern batch layout mismatch");
            count=a.shape(0);if(count<1)throw py::value_error("empty intern batch");
            schema+=py::str(a.dtype()).cast<std::string>()+":";for(int d=1;d<a.ndim();++d)schema+=std::to_string(a.shape(d))+",";schema+=';';
            widths.push_back(a.nbytes()/count);width+=widths.back();arrays.push_back(a);}
        if(count<0)throw py::value_error("intern fields required");
        Array<int64_t> ids(count);auto out=ids.mutable_data();
        {py::gil_scoped_release unlocked;for(ssize_t i=0;i<count;++i){std::string key=schema;key.resize(schema.size()+width);size_t at=schema.size();
            for(size_t j=0;j<arrays.size();++j){size_t bytes=widths[j];std::memcpy(key.data()+at,static_cast<const char*>(arrays[j].data())+i*bytes,bytes);at+=bytes;}
            auto found=entries_.find(key);if(found!=entries_.end()){out[i]=found->second.id;order_.splice(order_.end(),order_,found->second.order);continue;}
            out[i]=next_++;size_t cost=key.size()+256;if(cost>limit_)continue;
            while(bytes_+cost>limit_&&!order_.empty()){auto old=entries_.find(order_.front());bytes_-=old->second.cost;entries_.erase(old);order_.pop_front();}
            order_.push_back(key);entries_.emplace(key,Entry{out[i],cost,std::prev(order_.end())});bytes_+=cost;}}
        return ids;
    }
    size_t bytes()const{return bytes_;}
    void set_limit(size_t limit){limit_=limit;while(bytes_>limit_&&!order_.empty()){
        auto old=entries_.find(order_.front());bytes_-=old->second.cost;entries_.erase(old);order_.pop_front();}}
    void clear(){entries_.clear();order_.clear();bytes_=0;/* IDs remain unique across clears. */}
};

inline py::tuple pack_jobs(py::list roots,py::list worlds,Array<int64_t> slots,int stride,int offset) {
    int n=py::len(roots);if(py::len(worlds)!=size_t(n)||slots.size()!=n||stride<1)throw py::value_error("invalid admission arrays");
    std::vector<Array<uint32_t>> bits;std::vector<Array<int64_t>> seeds;std::vector<int> counts,draw;
    int total=0;for(int i=0;i<n;++i){auto root=roots[i].cast<py::dict>();Array<uint32_t> b(root["bits"]);Array<int64_t> w(worlds[i]);
        int count=b.size()*w.size();if(count<1||count>stride)throw py::value_error("invalid paired budget");
        counts.push_back(count);total+=count;bits.push_back(b);seeds.push_back(w);draw.push_back(root["draw_selection"].cast<int>());}
    Array<int64_t> source(total),destination(total),world(total),actions(total);Array<uint32_t> selected(total);Array<int32_t> drawing(total);
    auto s=source.mutable_data(),d=destination.mutable_data(),w=world.mutable_data(),a=actions.mutable_data();auto b=selected.mutable_data();auto dr=drawing.mutable_data();auto sl=slots.data();
    {py::gil_scoped_release unlocked;int at=0;for(int i=0;i<n;++i){int na=bits[i].size();for(int k=0;k<counts[i];++k,++at){
        s[at]=sl[i];d[at]=offset+sl[i]*stride+k;w[at]=seeds[i].data()[k/na];a[at]=k%na;b[at]=bits[i].data()[k%na];dr[at]=draw[i];}}}
    return py::make_tuple(source,destination,world,selected,drawing,actions,counts);
}

inline void validate_fork(Array<int64_t> sources,Array<int64_t> destinations,Array<bool> loaded,int64_t capacity) {
    if(sources.size()<1||sources.size()!=destinations.size()||loaded.size()!=capacity)throw py::value_error("invalid fork layout");
    auto src=sources.data(),dst=destinations.data();auto live=loaded.data();bool valid=true;
    {py::gil_scoped_release unlocked;std::unordered_set<int64_t> target;target.reserve(destinations.size());
        for(ssize_t i=0;i<sources.size();++i){if(src[i]<0||src[i]>=capacity||dst[i]<0||dst[i]>=capacity||!live[src[i]]||!target.insert(dst[i]).second){valid=false;break;}}
        if(valid)for(ssize_t i=0;i<sources.size();++i)if(target.count(src[i])){valid=false;break;}}
    if(!valid)throw py::value_error("fork sources must be live and disjoint from unique destinations");
}

inline Array<int64_t> completed_jobs(const std::vector<Handle>& handles,py::dict jobs) {
    std::vector<Array<int64_t>> arrays;ssize_t count=0;
    for(const auto& h:handles){Array<int64_t> a(jobs[py::int_(h.slot)].cast<py::tuple>()[0]);count+=a.size();arrays.push_back(a);}
    Array<int64_t> result(count);auto out=result.mutable_data();
    {py::gil_scoped_release unlocked;for(const auto& a:arrays){std::memcpy(out,a.data(),a.size()*sizeof(int64_t));out+=a.size();}}return result;
}

inline py::dict complete_results(const std::vector<Handle>& handles,py::dict bank,py::dict jobs,Array<double> values) {
    py::dict result;ssize_t offset=0;
    for(const auto& h:handles){auto row=bank[py::int_(h.slot)].cast<py::dict>();auto root=row["root"].cast<py::dict>();
        Array<int64_t> action(jobs[py::int_(h.slot)].cast<py::tuple>()[1]);int count=py::len(root["actions"]),worlds=py::len(row["worlds"]);
        if(offset+action.size()>values.size()||count<1||worlds<1)throw py::value_error("invalid completed root layout");
        Array<double> mean(count);Array<int64_t> visits(count);auto m=mean.mutable_data();auto v=visits.mutable_data();auto a=action.data();auto x=values.data()+offset;bool valid=true;
        {py::gil_scoped_release unlocked;std::fill(m,m+count,0.);std::fill(v,v+count,0);
            for(ssize_t i=0;i<action.size();++i){if(a[i]<0||a[i]>=count){valid=false;break;}m[a[i]]+=x[i];++v[a[i]];}
            for(int i=0;i<count;++i){valid&=v[i]==worlds;m[i]/=worlds;}}
        if(!valid)throw std::runtime_error("incomplete paired streaming teacher");
        py::dict out;out["actions"]=root["actions"];out["values"]=std::move(mean);out["visits"]=std::move(visits);result[h.key]=std::move(out);offset+=action.size();
    }
    if(offset!=values.size())throw py::value_error("extra completed values");return result;
}

inline void bind(py::module_& m) {
    m.attr("host_runtime_version")=2;
    py::class_<Handle>(m,"RootHandle").def_readonly("slot",&Handle::slot).def_readonly("generation",&Handle::generation).def_readonly("key",&Handle::key);
    py::class_<RootLedger>(m,"RootLedger").def(py::init<int>()).def("claim",&RootLedger::claim).def("release",&RootLedger::release)
        .def("set_deadlines",&RootLedger::set_deadlines).def("deadline",&RootLedger::deadline,py::call_guard<py::gil_scoped_release>())
        .def("completed",&RootLedger::completed).def_readonly("pending",&RootLedger::pending).def("__len__",[](RootLedger& x){return py::len(x.pending);});
    py::class_<Lease>(m,"PageLease").def_readonly("key",&Lease::key).def_readonly("serial",&Lease::serial).def_readonly("count",&Lease::count).def_readonly("pages",&Lease::pages);
    py::class_<DevicePages>(m,"DevicePages").def(py::init<int,int,int,py::function,int,int>(),py::arg("root_capacity"),py::arg("budget_cap"),py::arg("state_bytes"),py::arg("allocator"),py::arg("page_size")=16,py::arg("growth_pages")=256)
        .def("acquire",&DevicePages::acquire).def("release",&DevicePages::release).def("statistics",&DevicePages::statistics).def("close",&DevicePages::close);
    py::class_<ExactValueCache>(m,"ExactValueCache").def(py::init<int64_t>()).def("clear",&ExactValueCache::clear)
        .def_property_readonly("stats",&ExactValueCache::statistics).def("forward",&ExactValueCache::forward,py::arg("observations"),py::arg("forward"),py::kw_only(),py::arg("namespace")=py::tuple(),py::arg("indexed")=false);
    py::class_<ByteIntern>(m,"ByteIntern").def(py::init<size_t>()).def("intern",&ByteIntern::intern)
        .def("clear",&ByteIntern::clear).def("set_limit",&ByteIntern::set_limit).def_property_readonly("bytes",&ByteIntern::bytes);
    m.def("pack_arrays",&pack_arrays);m.def("pack_jobs",&pack_jobs);m.def("stack_fields",&stack_fields);
    m.def("pack_snapshots",&pack_snapshots);
    m.def("validate_fork",&validate_fork);m.def("completed_jobs",&completed_jobs);m.def("complete_results",&complete_results);
}
} // namespace resident_host
