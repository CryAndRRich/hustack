#include <map>
#include <set>

using namespace std;

class MaxStack {
    map<int, int> indexToValue;
    map<int, set<int>> valueToIndices;
    int counter = 0;

public:
    MaxStack() {}

    void push(int x) {
        indexToValue[counter] = x;
        valueToIndices[x].insert(counter);
        counter++;
    }

    int pop() {
        auto it = --indexToValue.end();
        int idx = it -> first, val = it -> second;
        indexToValue.erase(it);
        valueToIndices[val].erase(idx);
        if (valueToIndices[val].empty())
            valueToIndices.erase(val);
        return val;
    }

    int top() {
        return indexToValue.rbegin() -> second;
    }

    int peekMax() {
        return valueToIndices.rbegin() -> first;
    }

    int popMax() {
        auto it = --valueToIndices.end();
        int val = it -> first;
        auto& indices = it -> second;
        int idx = *indices.rbegin();
        indices.erase(idx);
        if (indices.empty())
            valueToIndices.erase(val);
        indexToValue.erase(idx);
        return val;
    }
};

/**
 * Your MaxStack object will be instantiated and called as such:
 * MaxStack* obj = new MaxStack();
 * obj -> push(x);
 * int param_2 = obj -> pop();
 * int param_3 = obj -> top();
 * int param_4 = obj -> peekMax();
 * int param_5 = obj -> popMax();
 */