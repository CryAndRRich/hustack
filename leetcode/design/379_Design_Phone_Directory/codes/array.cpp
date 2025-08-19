#include <vector>
using namespace std;

class PhoneDirectory {
    vector<bool> isSlotAvailable;

public:
    PhoneDirectory(int maxNumbers) {
        isSlotAvailable = vector<bool>(maxNumbers, true);
    }

    int get() {
        for (int i = 0; i < (int)isSlotAvailable.size(); ++i) {
            if (isSlotAvailable[i]) {
                isSlotAvailable[i] = false;
                return i;
            }
        }
        return -1;
    }

    bool check(int number) {
        return isSlotAvailable[number];
    }

    void release(int number) {
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