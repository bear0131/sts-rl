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

pybind11_add_module(slaythespire bindings/slaythespire.cpp bindings/bindings-util.cpp ${sts_lightspeed_SOURCES})
if (STS_ENABLE_RULE_TRACE)
    target_compile_definitions(slaythespire PRIVATE STS_ENABLE_RULE_TRACE)
endif()
target_include_directories(slaythespire PUBLIC include)
target_include_directories(slaythespire PUBLIC json/single_include)
target_include_directories(slaythespire PUBLIC bindings)


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

# Internal pool boundaries are deliberately not exposed through the Python API.
# pytest builds this isolated regression executable only when those tests run.
add_executable(relic-pool-contract EXCLUDE_FROM_ALL
        ../../tests/integration/native_relic_pool.cpp ${sts_lightspeed_SOURCES})
target_include_directories(relic-pool-contract PRIVATE include json/single_include)
if (STS_ENABLE_RULE_TRACE)
    target_compile_definitions(relic-pool-contract PRIVATE STS_ENABLE_RULE_TRACE)
endif()
