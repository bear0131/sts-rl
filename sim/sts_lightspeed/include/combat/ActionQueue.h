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
#include <memory>
#include <vector>


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

    private:
        // Ordinary search states retain inline storage. Only a long callback
        // chain allocates a larger ring; no pending action may be overwritten.
        std::unique_ptr<std::vector<Action>> overflow;
        void grow();
        ActionFunction &callback(int slot) {
            return overflow ? (*overflow)[slot].actFunc : arr[slot];
        }
        const ActionFunction &callback(int slot) const {
            return overflow ? (*overflow)[slot].actFunc : arr[slot];
        }
        bool clearsOnVictory(int slot) const {
            return overflow ? (*overflow)[slot].clearOnCombatVictory : bits[slot];
        }
        void setClearFlag(int slot, bool value) {
            if (overflow) (*overflow)[slot].clearOnCombatVictory = value;
            else bits.set(slot, value);
        }
    };

    template<int capacity>
    void ActionQueue<capacity>::grow() {
        const int oldCapacity = getCapacity();
        auto expanded = std::make_unique<std::vector<Action>>(oldCapacity * 2);
        for (int i = 0, slot = front; i < size; ++i) {
            (*expanded)[i] = Action(std::move(callback(slot)), clearsOnVictory(slot));
            callback(slot) = nullptr;
            if (++slot == oldCapacity) slot = 0;
        }
        overflow = std::move(expanded);
        bits.reset();
        front = 0;
        back = size;
    }

    template<int capacity>
    ActionQueue<capacity> &ActionQueue<capacity>::operator=(const ActionQueue &rhs) {
        if (this == &rhs) return *this;
        clear();
        if (!rhs.overflow) front = back = rhs.front;
        const int bound = rhs.getCapacity();
        for (int i = 0, slot = rhs.front; i < rhs.size; ++i) {
            pushBack(Action(rhs.callback(slot), rhs.clearsOnVictory(slot)));
            if (++slot == bound) slot = 0;
        }
        return *this;
    }

    template<int capacity>
    void ActionQueue<capacity>::clear() {
        if (overflow) {
            overflow.reset();
            size = 0;
        }
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
        const int bound = getCapacity();
        for (int i = 0; i < size; ++i) {
            if (clearsOnVictory(read)) {
                callback(read) = nullptr;
                setClearFlag(read, false);
            } else {
                if (write != read) {
                    callback(write) = std::move(callback(read));
                    callback(read) = nullptr;
                }
                setClearFlag(write, false);
                if (++write == bound) write = 0;
                ++kept;
            }
            if (++read == bound) read = 0;
        }
        size = kept;
        back = write;
    }

    template <int capacity>
    void ActionQueue<capacity>::pushFront(Action a) {
        if (size == getCapacity()) grow();
        --front;
        ++size;
        if (front < 0) {
            front = getCapacity()-1;
        }
        setClearFlag(front, a.clearOnCombatVictory);
        callback(front) = std::move(a.actFunc);
    }

    template<int capacity>
    void ActionQueue<capacity>::pushBack(Action a) {
        if (size == getCapacity()) grow();
        if (back >= getCapacity()) {
            back = 0;
        }
        setClearFlag(back, a.clearOnCombatVictory);
        callback(back) = std::move(a.actFunc);
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
        ActionFunction a = std::move(callback(front));
        callback(front) = nullptr;
        setClearFlag(front, false);
        ++front;
        --size;
        if (front >= getCapacity()) {
            front = 0;
        }
        return a;
    }

    template<int capacity>
    int ActionQueue<capacity>::getCapacity() const {
        return overflow ? static_cast<int>(overflow->size()) : capacity;
    }

}


#endif //STS_LIGHTSPEED_ACTIONQUEUE_H
