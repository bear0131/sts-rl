//
// Created by gamerpuppy on 7/4/2021.
//

#ifndef STS_LIGHTSPEED_ACTIONQUEUE_H
#define STS_LIGHTSPEED_ACTIONQUEUE_H

#include "sts_common.h"

#include <bitset>
#include <array>
#include <functional>
#include <cassert>
#include <utility>


namespace sts {

    class BattleContext;
    typedef std::function<void (BattleContext &)> ActionFunction;

    struct Action {
        ActionFunction actFunc;
        bool clearOnCombatVictory = true;

        Action() = default;
        Action(ActionFunction a) : actFunc(std::move(a)) {}
        Action(ActionFunction a, bool b) : actFunc(std::move(a)), clearOnCombatVictory(b) {}
    };


    // Simple deque
    template<int capacity>
    struct ActionQueue {
        friend BattleContext;
        int front = 0;
        int back = 0;
        int size = 0;
        std::bitset<capacity> bits; // for the shouldClear field

#ifdef sts_action_queue_use_raw_array
        ActionFunction arr[capacity];
#else
        std::array<ActionFunction,capacity> arr;
#endif
        ActionQueue() = default;
        // A search clone owns only pending callbacks, never retired slot storage.
        ActionQueue(const ActionQueue &rhs) { *this = rhs; }
        ActionQueue &operator=(const ActionQueue &rhs);

        void clear();
        void clearOnCombatVictory();
        void pushFront(Action a);
        void pushBack(Action a);
        bool isEmpty();
        ActionFunction popFront();
        [[nodiscard]] int getCapacity() const;
    };

    template<int capacity>
    ActionQueue<capacity> &ActionQueue<capacity>::operator=(const ActionQueue &rhs) {
        if (this == &rhs) return *this;
        clear();
        front = back = rhs.front;
        for (int i = 0, slot = rhs.front; i < rhs.size; ++i) {
            pushBack(Action(rhs.arr[slot], rhs.bits[slot]));
            if (++slot == capacity) slot = 0;
        }
        return *this;
    }

    template<int capacity>
    void ActionQueue<capacity>::clear() {
        while (size > 0) {
            arr[front] = nullptr;
            if (++front == capacity) front = 0;
            --size;
        }
        back = 0;
        front = 0;
        bits.reset();
    }

    template<int capacity>
    void ActionQueue<capacity>::clearOnCombatVictory() {
        // Keep survivor order and flags even across the ring's physical end.
        int read = front, write = front, kept = 0;
        for (int i = 0; i < size; ++i) {
            if (bits[read]) {
                arr[read] = nullptr;
                bits.reset(read);
            } else {
                if (write != read) {
                    arr[write] = std::move(arr[read]);
                    arr[read] = nullptr;
                }
                bits.reset(write);
                if (++write == capacity) write = 0;
                ++kept;
            }
            if (++read == capacity) read = 0;
        }
        size = kept;
        back = write;
    }

    template <int capacity>
    void ActionQueue<capacity>::pushFront(Action a) {
#ifdef sts_asserts
        assert(size != capacity);
#endif
        --front;
        ++size;
        if (front < 0) {
            front = capacity-1;
        }
        bits.set(front, a.clearOnCombatVictory);
        arr[front] = std::move(a.actFunc);
    }

    template<int capacity>
    void ActionQueue<capacity>::pushBack(Action a) {
#ifdef sts_asserts
        if (size >= capacity) {
            assert(false);
        }
#endif
        if (back >= capacity) {
            back = 0;
        }
        bits.set(back, a.clearOnCombatVictory);
        arr[back] = std::move(a.actFunc);
        ++back;
        ++size;
    }

    template<int capacity>
    bool ActionQueue<capacity>::isEmpty() {
        return size == 0;
    }

    template<int capacity>
    ActionFunction ActionQueue<capacity>::popFront() {
#ifdef sts_asserts
        assert(size > 0 );
#endif
        ActionFunction a = std::move(arr[front]);
        arr[front] = nullptr;
        bits.reset(front);
        ++front;
        --size;
        if (front >= capacity) {
            front = 0;
        }
        return a;
    }

    template<int capacity>
    int ActionQueue<capacity>::getCapacity() const {
        return capacity;
    }

}


#endif //STS_LIGHTSPEED_ACTIONQUEUE_H
