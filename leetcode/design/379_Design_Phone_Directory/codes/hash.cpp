#include <unordered_set>
using namespace std;

class PhoneDirectory {
    unordered_set<int> slotsAvailable;

public:
    PhoneDirectory(int maxNumbers) {
        for (int i = 0; i < maxNumbers; ++i)
            slotsAvailable.insert(i);
    }

    int get() {
        if (slotsAvailable.empty()) return -1;
        int slot = *slotsAvailable.begin();
        slotsAvailable.erase(slot);
        return slot;
    }

    bool check(int number) {
        return slotsAvailable.count(number) > 0;
    }

    void release(int number) {
        slotsAvailable.insert(number);
    }
};

/**
 * Your PhoneDirectory object will be instantiated and called as such:
 * PhoneDirectory* obj = new PhoneDirectory(maxNumbers);
 * int param_1 = obj->get();
 * bool param_2 = obj->check(number);
 * obj->release(number);
 */