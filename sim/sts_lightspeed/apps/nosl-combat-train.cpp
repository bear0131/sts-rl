#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "sim/search/NoslCombatTrainer.h"

using namespace sts;
using namespace sts::search;

namespace {

struct Options {
    int episodes = 10;
    int simulations = 128;
    int depth = 3;
    int maxDecisions = 500;
    int ascension = 0;
    int threads = 1;
    std::uint64_t seed = 1;
    MonsterEncounter encounter = MonsterEncounter::JAW_WORM;
    bool train = true;
    std::string dumpPath;
};

struct WorkerStats {
    std::uint64_t decisions = 0;
    std::uint64_t finalHp = 0;
    std::uint64_t wins = 0;
    std::uint64_t fallbacks = 0;
    int episodes = 0;
    std::size_t policyStates = 0;
    std::size_t valueStates = 0;
    NoslCombatSearchStats search;
};

void usage(const char *program) {
    std::cout << "usage: " << program
              << " [--episodes N] [--simulations N] [--depth N] [--max-decisions N]"
              << " [--seed N] [--ascension N] [--encounter NAME|ID] [--train 0|1]"
              << " [--dump-policy PATH] [--threads N]\n";
}

bool parseInt(const char *value, int &out) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parseUint64(const char *value, std::uint64_t &out) {
    char *end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint64_t>(parsed);
    return true;
}

MonsterEncounter parseEncounter(const std::string &name) {
    char *end = nullptr;
    const long parsed = std::strtol(name.c_str(), &end, 10);
    if (end && *end == '\0' && parsed >= 0 &&
        parsed <= static_cast<long>(MonsterEncounter::MYSTERIOUS_SPHERE_EVENT)) {
        return static_cast<MonsterEncounter>(parsed);
    }
    for (int i = 0; i <= static_cast<int>(MonsterEncounter::MYSTERIOUS_SPHERE_EVENT); ++i) {
        if (name == monsterEncounterEnumNames[i]) {
            return static_cast<MonsterEncounter>(i);
        }
    }
    return MonsterEncounter::JAW_WORM;
}

bool parseOptions(int argc, const char **argv, Options &options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return false;
        }
        if (i + 1 >= argc) {
            return false;
        }
        const char *value = argv[++i];
        bool recognized = true;
        if (arg == "--episodes") recognized = parseInt(value, options.episodes);
        else if (arg == "--simulations") recognized = parseInt(value, options.simulations);
        else if (arg == "--depth") recognized = parseInt(value, options.depth);
        else if (arg == "--max-decisions") recognized = parseInt(value, options.maxDecisions);
        else if (arg == "--ascension") recognized = parseInt(value, options.ascension);
        else if (arg == "--threads") recognized = parseInt(value, options.threads);
        else if (arg == "--seed") recognized = parseUint64(value, options.seed);
        else if (arg == "--encounter") options.encounter = parseEncounter(value);
        else if (arg == "--train") {
            int train = 0;
            recognized = parseInt(value, train);
            options.train = train != 0;
        } else if (arg == "--dump-policy") options.dumpPath = value;
        else recognized = false;
        if (!recognized) {
            std::cerr << "invalid option or value: " << arg << " " << value << '\n';
            return false;
        }
    }
    return true;
}

}

int main(int argc, const char **argv) {
    Options options;
    if (!parseOptions(argc, argv, options)) {
        usage(argv[0]);
        return argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") ? 0 : 2;
    }
    if (options.episodes < 1 || options.simulations < 1 || options.depth < 1 ||
        options.maxDecisions < 1 || options.threads < 1) {
        std::cerr << "episodes, simulations, depth, max-decisions, and threads must be positive\n";
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const int threadCount = std::min(options.threads, options.episodes);
    std::atomic<int> nextEpisode{0};
    std::atomic<bool> dumpFailed{false};
    std::mutex outputMutex;
    std::vector<WorkerStats> workers(threadCount);

    auto runWorker = [&](int workerIdx) {
        NoslCombatSearchConfig config;
        config.simulations = options.simulations;
        config.maxDepth = options.depth;
        config.seed = options.seed + static_cast<std::uint64_t>(workerIdx);
        NoslCombatTrainer trainer(config);
        auto &worker = workers[workerIdx];

        while (true) {
            const int episodeIdx = nextEpisode.fetch_add(1, std::memory_order_relaxed);
            if (episodeIdx >= options.episodes) {
                break;
            }
            const auto episode = trainer.runEpisode(
                options.seed + static_cast<std::uint64_t>(episodeIdx),
                CharacterClass::IRONCLAD,
                options.ascension,
                options.encounter,
                options.maxDecisions,
                options.train
            );
            ++worker.episodes;
            worker.decisions += episode.decisions;
            worker.finalHp += episode.finalHp;
            worker.wins += episode.outcome == Outcome::PLAYER_VICTORY;
            worker.fallbacks += episode.fallbackActions;
            worker.search.simulations += episode.search.simulations;
            worker.search.transitions += episode.search.transitions;
            worker.search.expandedNodes += episode.search.expandedNodes;
            std::scoped_lock lock(outputMutex);
            std::cout << "episode=" << episodeIdx
                      << " worker=" << workerIdx
                      << " decisions=" << episode.decisions
                      << " outcome=" << static_cast<int>(episode.outcome)
                      << " final_hp=" << episode.finalHp
                      << " simulations=" << episode.search.simulations
                      << " transitions=" << episode.search.transitions
                      << " fallbacks=" << episode.fallbackActions << '\n';
        }
        worker.policyStates = trainer.policyStateCount();
        worker.valueStates = trainer.valueStateCount();
        if (!options.dumpPath.empty()) {
            const std::string path = threadCount == 1
                ? options.dumpPath
                : options.dumpPath + ".worker-" + std::to_string(workerIdx);
            if (!trainer.saveTable(path)) {
                dumpFailed.store(true, std::memory_order_relaxed);
            }
        }
    };

    if (threadCount == 1) {
        runWorker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(threadCount);
        for (int workerIdx = 0; workerIdx < threadCount; ++workerIdx) {
            threads.emplace_back(runWorker, workerIdx);
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }

    WorkerStats total;
    for (const auto &worker : workers) {
        total.episodes += worker.episodes;
        total.decisions += worker.decisions;
        total.finalHp += worker.finalHp;
        total.wins += worker.wins;
        total.fallbacks += worker.fallbacks;
        total.policyStates += worker.policyStates;
        total.valueStates += worker.valueStates;
        total.search.simulations += worker.search.simulations;
        total.search.transitions += worker.search.transitions;
        total.search.expandedNodes += worker.search.expandedNodes;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double safeSeconds = std::max(seconds, 1e-9);
    std::cout << "episodes=" << total.episodes
              << " encounter=" << monsterEncounterEnumNames[static_cast<int>(options.encounter)]
              << " simulations_per_root=" << options.simulations
              << " depth=" << options.depth
              << " threads=" << threadCount
              << " train=" << (options.train ? 1 : 0)
              << " decisions=" << total.decisions
              << " simulations=" << total.search.simulations
              << " transitions=" << total.search.transitions
              << " expanded_nodes=" << total.search.expandedNodes
              << " wins=" << total.wins
              << " fallbacks=" << total.fallbacks
              << " avg_final_hp=" << (total.episodes ? static_cast<double>(total.finalHp) / total.episodes : 0.0)
              << " elapsed_s=" << seconds
              << " decisions_per_s=" << total.decisions / safeSeconds
              << " simulations_per_s=" << total.search.simulations / safeSeconds
              << " transitions_per_s=" << total.search.transitions / safeSeconds
              << " avg_ms_per_root=" << (total.decisions ? seconds * 1000.0 / total.decisions : 0.0)
              << " policy_states=" << total.policyStates
              << " value_states=" << total.valueStates
              << " training_shards=" << threadCount << '\n';

    if (dumpFailed.load(std::memory_order_relaxed)) {
        std::cerr << "failed to save policy table: " << options.dumpPath << '\n';
        return 1;
    }
    return 0;
}
