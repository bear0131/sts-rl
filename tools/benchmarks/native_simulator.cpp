// Native-only throughput: no Python calls, observation encoding or model inference.
// Build against both revisions with identical compiler flags; compare work/digest.
#include "sim/search/NoslCombatTrainer.h"
#include <chrono>
#include <atomic>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace sts;
using Clock = std::chrono::steady_clock;

static std::uint64_t mix(std::uint64_t &rng) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

static void digest(std::uint64_t &value, std::uint64_t item) {
    value = (value ^ item) * 1099511628211ULL;
}

static thread_local volatile std::uint64_t cloneSink = 0;
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void consume(const BattleContext &state) {
    cloneSink += state.player.curHp + state.cards.cardsInHand + state.turn;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "mode(clone|playout|search) iterations seed timeout_seconds [threads]\n";
        return 2;
    }
    try {
        const std::string mode = argv[1];
        const int iterations = std::stoi(argv[2]);
        const auto seed = std::stoull(argv[3]);
        const double timeout = std::stod(argv[4]);
        const int threads = argc == 6 ? std::stoi(argv[5]) : 1;
        if (iterations < 1 || iterations > 10000000 || timeout <= 0 || timeout > 600)
            throw std::invalid_argument("bounded iterations/timeout required");
        if (threads < 1 || threads > 32 || threads > iterations)
            throw std::invalid_argument("threads must be 1..32 and no more than iterations");
        if (mode != "clone" && mode != "playout" && mode != "search")
            throw std::invalid_argument("unknown workload");
        const MonsterEncounter encounters[] = {MonsterEncounter::CULTIST,
            MonsterEncounter::JAW_WORM, MonsterEncounter::TWO_LOUSE,
            MonsterEncounter::GREMLIN_NOB, MonsterEncounter::THREE_SENTRIES,
            MonsterEncounter::HEXAGHOST};
        const auto setupStart = Clock::now();
        std::vector<BattleContext> roots;
        std::uint64_t populatedSlots = 0, pendingActions = 0;
        // Include starts and mid-fight roots whose queues have already drained.
        for (int i = 0; i < 96; ++i) {
            GameContext game(CharacterClass::IRONCLAD, seed + i, 0);
            game.floorNum = 1;
            BattleContext state;
            state.init(game, encounters[i % 6]);
            std::uint64_t rng = seed + i + 1;
            for (int j = 0; j < i % 9 && state.outcome == Outcome::UNDECIDED; ++j) {
                auto actions = search::Action::enumerateCombatActions(state);
                if (actions.empty()) break;
                actions[mix(rng) % actions.size()].execute(state);
            }
            if (state.outcome != Outcome::UNDECIDED) continue;
            for (const auto &action : state.actionQueue.arr) populatedSlots += bool(action);
            pendingActions += state.actionQueue.size;
            roots.push_back(state);
        }
        const double setup = std::chrono::duration<double>(Clock::now() - setupStart).count();
        // Per-action counters must not introduce false sharing into the benchmark.
        struct alignas(64) Result {
            std::uint64_t checksum = 1469598103934665603ULL, actions = 0, simulations = 0;
            std::uint64_t wins = 0, capped = 0, sink = 0;
            std::exception_ptr error;
        };
        std::vector<Result> results(threads);
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        Clock::time_point start;
        const auto run = [&](int worker) {
            auto &result = results[worker];
            auto &checksum = result.checksum;
            auto &actions = result.actions;
            auto &simulations = result.simulations;
            auto &wins = result.wins;
            auto &capped = result.capped;
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            try {
                cloneSink = 0;
                search::NoslCombatSearchConfig config;
                config.simulations = 128; config.maxDepth = 8; config.fixedWorldline = true;
                search::NoslCombatTrainer trainer(config);
                for (int i = worker, local = 0; i < iterations; i += threads, ++local) {
                    if ((local & 127) == 0 && std::chrono::duration<double>(Clock::now() - start).count() > timeout)
                        throw std::runtime_error("benchmark exceeded time budget");
                    const auto &root = roots[i % roots.size()];
                    if (mode == "clone") {
                        BattleContext copy(root);
                        consume(copy);
                    } else if (mode == "playout") {
                        BattleContext state(root);
                        std::uint64_t rng = seed + i + 1;
                        int step = 0;
                        for (; step < 256 && state.outcome == Outcome::UNDECIDED; ++step) {
                            auto legal = search::Action::enumerateCombatActions(state);
                            if (legal.empty()) throw std::runtime_error("nonterminal without legal action");
                            const auto action = legal[mix(rng) % legal.size()];
                            action.execute(state);
                            digest(checksum, action.bits); ++actions;
                        }
                        capped += state.outcome == Outcome::UNDECIDED;
                        wins += state.outcome == Outcome::PLAYER_VICTORY;
                        digest(checksum, search::NoslCombatTrainer::stateKey(state));
                    } else {
                        search::NoslCombatSearchStats stats;
                        const auto decision = trainer.search(root, seed + i, &stats);
                        actions += stats.transitions; simulations += stats.simulations;
                        digest(checksum, decision.action.bits);
                        for (const auto &visit : decision.rootVisits) {
                            digest(checksum, visit.first); digest(checksum, visit.second);
                        }
                    }
                }
                result.sink = cloneSink;
            } catch (...) { result.error = std::current_exception(); }
        };
        std::vector<std::thread> workers;
        // Thread creation is setup; trainer construction and joins are timed.
        try {
            if (threads > 1)
                for (int worker = 0; worker < threads; ++worker) workers.emplace_back(run, worker);
        } catch (...) {
            start = Clock::now();
            go.store(true, std::memory_order_release);
            for (auto &worker : workers) worker.join();
            throw;
        }
        if (threads > 1)
            while (ready.load(std::memory_order_acquire) != threads) std::this_thread::yield();
        start = Clock::now();
        go.store(true, std::memory_order_release);
        if (threads == 1) run(0);
        for (auto &worker : workers) worker.join();
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        std::uint64_t checksum = results[0].checksum, actions = 0, simulations = 0;
        std::uint64_t wins = 0, capped = 0, sink = 0;
        for (int worker = 0; worker < threads; ++worker) {
            const auto &result = results[worker];
            if (result.error) std::rethrow_exception(result.error);
            if (worker) digest(checksum, result.checksum);
            actions += result.actions; simulations += result.simulations;
            wins += result.wins; capped += result.capped; sink += result.sink;
        }
        std::cout << std::setprecision(12) << "{\"mode\":\"" << mode
            << "\",\"iterations\":" << iterations << ",\"seed\":" << seed
            << ",\"threads\":" << threads
            << ",\"roots\":" << roots.size() << ",\"root_populated_callback_slots\":" << populatedSlots
            << ",\"root_pending_actions\":" << pendingActions
            << ",\"battle_bytes\":" << sizeof(BattleContext)
            << ",\"setup_seconds\":" << setup << ",\"seconds\":" << elapsed
            << ",\"actions\":" << actions << ",\"simulations\":" << simulations
            << ",\"wins\":" << wins << ",\"capped\":" << capped
            << ",\"digest\":" << checksum << ",\"clone_sink\":" << sink << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
