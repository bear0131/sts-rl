// Build against the explicit native simulator and its logical-state codec.
// CPU serialization owns its capacity; the fixed CUDA import buffer is a
// transport limit, never a limit on legal generated cards in the live game.
#include "combat/BattleContext.h"
#include "state_codec.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct HostSnapshotWriter {
    static constexpr bool reading = false;
    std::vector<unsigned long long> words;
    unsigned long long max_vector = 0;
    HostSnapshotWriter() { words.reserve(8192); }
    void word(unsigned long long& value) { words.push_back(value); }
    template<class... T> void fields(T&... value) { (field(value), ...); }
    template<class T> void field(T& value) {
        if constexpr (__is_class(T) || device_state::is_array<T>::value) {
            device_state::visit(*this, value);
        } else {
            // Retain the canonical scalar size/bit encoding, including enums.
            unsigned long long scalar[2]{};
            device_state::Archive<false> writer{scalar, 2};
            writer.field(value);
            for (int i = 0; i < writer.at; ++i) word(scalar[i]);
        }
    }
};

PYBIND11_MODULE(deck_cuda_bridge, module) {
    module.attr("cpu_battle_bytes") = sizeof(sts::BattleContext);
    module.def("snapshot", [](const sts::BattleContext& source) {
        auto state = source;
#ifdef STS_ENABLE_RULE_TRACE
        state.atomicStepObserver = {};
#endif
        HostSnapshotWriter writer;
        writer.field(state);
        py::array_t<unsigned long long> words(writer.words.size());
        std::copy(writer.words.begin(), writer.words.end(), words.mutable_data());
        py::dict result;
        result["words"] = words;
        result["queue_size"] = state.actionQueue.size;
        result["max_vector"] = writer.max_vector;
        result["portable"] = state.actionQueue.size == 0 && writer.max_vector <= 128
                             && writer.words.size() <= 8192;
        result["input_state"] = int(state.inputState);
        result["outcome"] = int(state.outcome);
        return result;
    });
}
