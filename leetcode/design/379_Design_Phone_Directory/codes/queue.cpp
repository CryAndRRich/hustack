#include <vector>
#include <queue>
using namespace std;

class PhoneDirectory {
    queue<int> slotsAvailableQueue;
    vector<bool> isSlotAvailable;

public:
    PhoneDirectory(int maxNumbers) {
        isSlotAvailable = vector<bool>(maxNumbers, true);
        for (int i = 0; i < maxNumbers; ++i) {
            slotsAvailableQueue.push(i);
        }
    }

    int get() {
        if (slotsAvailableQueue.empty()) {
            return -1;
        }
        int slot = slotsAvailableQueue.front();
        slotsAvailableQueue.pop();
        isSlotAvailable[slot] = false;
        return slot;
    }

    bool check(int number) {
        return isSlotAvailable[number];
    }

    void release(int number) {
        if (isSlotAvailable[number]) {
            return;
        }
        slotsAvailableQueue.push(number);
        isSlotAvailable[number] = true;
    }
};

/**
 * Your PhoneDirectory object will be instantiated and called as such:
 * PhoneDirectory* obj = new PhoneDirectory(maxNumbers);
 * int param_1 = obj->get();
 * bool param_2 = obj->check(number);
 * obj->release(number);
 */