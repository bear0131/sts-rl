#pragma once
#include <pybind11/numpy.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Immutable layout shared by owned rows, never by mutable game state. Numeric
// storage stays in C++ from settlement through cache lookup and miss batching.
namespace upper_observation {
namespace py = pybind11;
struct Field {
    std::string name;
    std::vector<py::ssize_t> shape;
    std::size_t offset, bytes, size;
    bool floating;
};
struct Layout {
    std::vector<Field> fields;
    std::unordered_map<std::string,std::size_t> indices;
    std::size_t bytes=0;
    explicit Layout(py::dict prototypes) {
        for(auto item:prototypes) {
            auto a=py::cast<py::array>(item.second);
            const bool floating=a.dtype().equal(py::dtype::of<float>());
            if(a.ndim()!=1 || a.size()<1 ||
                    (!floating && !a.dtype().equal(py::dtype::of<std::int64_t>())))
                throw std::invalid_argument("upper reward observation prototype layout mismatch");
            bytes=(bytes+7)&~std::size_t(7);
            auto name=py::cast<std::string>(item.first);
            indices.emplace(name,fields.size());
            fields.push_back({std::move(name),{a.shape(0)},bytes,
                static_cast<std::size_t>(a.nbytes()),static_cast<std::size_t>(a.size()),floating});
            bytes+=a.nbytes();
        }
    }
    const Field &at(const std::string &name) const {
        auto found=indices.find(name);
        if(found==indices.end()) throw py::key_error(name);
        return fields[found->second];
    }
};
struct Row : std::enable_shared_from_this<Row> {
    const std::shared_ptr<const Layout> layout;
    std::vector<std::uint64_t> storage;
    explicit Row(std::shared_ptr<const Layout> source)
        :layout(std::move(source)),storage((layout->bytes+7)/8,0) {}
    const char *data(const Field &field) const {
        return reinterpret_cast<const char *>(storage.data())+field.offset;
    }
    char *data(const Field &field) {
        return reinterpret_cast<char *>(storage.data())+field.offset;
    }
    py::array view(const std::string &name) {
        const auto &field=layout->at(name);
        auto dtype=field.floating?py::dtype::of<float>():py::dtype::of<std::int64_t>();
        // A view owns this row even after the encoder, source GC and Python
        // row wrapper are gone. Views remain independently mutable per row.
        return py::array(dtype,field.shape,{},data(field),py::cast(shared_from_this()));
    }
    py::list keys() const {
        py::list result;for(const auto &field:layout->fields) result.append(field.name);
        return result;
    }
    py::dict asDict() {
        py::dict result;for(const auto &field:layout->fields) result[py::str(field.name)]=view(field.name);
        return result;
    }
};
}
