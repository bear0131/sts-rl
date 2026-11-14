cmake_minimum_required(VERSION 3.19)
project(
        sts_lightspeed
        VERSION 0.1
        DESCRIPTION "High Performance, rng accurate implementation of Slay the Spire with tools for tree search and ml."
        LANGUAGES CXX)

option(STS_ENABLE_RULE_TRACE "Enable atomic combat snapshots for rule-data collection" ON)

file (GLOB_RECURSE sts_lightspeed_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS  "-Wno-shift-count-overflow -O3")

add_subdirectory(json)
add_subdirectory(pybind11)

find_package(Threads REQUIRED)
pybind11_add_module(slaythespire bindings/slaythespire.cpp bindings/bindings-util.cpp bindings/neural-observation.cpp bindings/upper-value.cpp bindings/small-playout.cpp ${sts_lightspeed_SOURCES})
# Rule-search priorities mirror Python doubles: contraction changes exact ties.
target_compile_options(slaythespire PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off>)
if (STS_ENABLE_RULE_TRACE)
    target_compile_definitions(slaythespire PRIVATE STS_ENABLE_RULE_TRACE)
endif()
target_include_directories(slaythespire PUBLIC include)
target_include_directories(slaythespire PUBLIC json/single_include)
target_include_directories(slaythespire PUBLIC bindings)

# The qualified full480 CPU fallback keeps its original WaveDiag owner/loop.
# Build against the same simulator headers and trace ABI as this extension.
option(STS_BUILD_FULL480 "Build the original full480 CPU fallback consumer" OFF)
if(STS_BUILD_FULL480)
    set(full480_root "${CMAKE_CURRENT_SOURCE_DIR}/../../rl/search/full480/native")
    set(full480_sources ${sts_lightspeed_SOURCES})
    list(FILTER full480_sources EXCLUDE REGEX "/SmallPlayout\\.cpp$")
    pybind11_add_module(_full480_cpu "${full480_root}/wave.cpp"
        "${full480_root}/bindings.cpp" ${full480_sources})
    target_include_directories(_full480_cpu PRIVATE "${full480_root}/include"
        "${full480_root}" include bindings json/single_include /usr/local/cuda/include)
    target_compile_definitions(_full480_cpu PRIVATE STS_FULL480_STANDALONE)
    if(STS_ENABLE_RULE_TRACE)
        target_compile_definitions(_full480_cpu PRIVATE STS_ENABLE_RULE_TRACE)
    endif()
    target_compile_options(_full480_cpu PRIVATE -ffp-contract=off -mavx2)
    find_library(full480_cuda_driver NAMES cuda cuda.so.1 REQUIRED)
    target_link_libraries(_full480_cpu PRIVATE Threads::Threads ${full480_cuda_driver})
endif()


add_executable(main apps/main.cpp ${sts_lightspeed_SOURCES})
target_link_directories(main PRIVATE json::nlohmann_json)
target_include_directories(main PUBLIC include)
target_include_directories(main PUBLIC json/include)

add_executable(test apps/test.cpp ${sts_lightspeed_SOURCES})
target_include_directories(test PUBLIC include)
target_link_directories(test PUBLIC json::nlohmann_json)
target_include_directories(test PUBLIC json/include)

add_executable(small-test apps/small-test.cpp bindings/bindings-util.cpp ${sts_lightspeed_SOURCES})
target_link_directories(small-test PRIVATE json::nlohmann_json)
target_include_directories(small-test PUBLIC include)
target_include_directories(small-test PUBLIC json/include)
target_include_directories(small-test PUBLIC bindings)

# Native NOSL combat search/training prototype. This target intentionally has
# no Python or Torch dependency, so it can be benchmarked on CPU workers.
add_executable(nosl-combat-train apps/nosl-combat-train.cpp ${sts_lightspeed_SOURCES})
target_link_directories(nosl-combat-train PRIVATE json::nlohmann_json)
target_include_directories(nosl-combat-train PUBLIC include)
target_include_directories(nosl-combat-train PUBLIC json/include)

# Internal state and pending callbacks are not exposed through the Python API.
# pytest builds this isolated regression executable only when those tests run.
add_executable(game-context-contract EXCLUDE_FROM_ALL
        ../../tests/integration/native_game_context.cpp ${sts_lightspeed_SOURCES})
target_include_directories(game-context-contract PRIVATE include json/single_include)
if (STS_ENABLE_RULE_TRACE)
    target_compile_definitions(game-context-contract PRIVATE STS_ENABLE_RULE_TRACE)
endif()

# Pure engine measurements avoid Python's per-action binding/encoding overhead.
add_executable(simulator-benchmark EXCLUDE_FROM_ALL
        ../../tools/benchmarks/native_simulator.cpp ${sts_lightspeed_SOURCES})
target_include_directories(simulator-benchmark PRIVATE include json/single_include)
target_compile_options(simulator-benchmark PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off>)
if (STS_ENABLE_RULE_TRACE)
    target_compile_definitions(simulator-benchmark PRIVATE STS_ENABLE_RULE_TRACE)
endif()

foreach(target slaythespire main test small-test nosl-combat-train game-context-contract simulator-benchmark)
    target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()
