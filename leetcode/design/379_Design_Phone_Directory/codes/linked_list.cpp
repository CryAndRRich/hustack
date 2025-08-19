#include <vector>
using namespace std;

struct Node {
    int val;
    Node* next;
    Node(int v) : val(v), next(nullptr) {}
};

class PhoneDirectory {
    vector<bool> isSlotAvailable;
    Node* head;

public:
    PhoneDirectory(int maxNumbers) {
        isSlotAvailable = vector<bool>(maxNumbers, true);
        head = new Node(-1);
        Node* cur = head;
        for (int i = 0; i < maxNumbers; ++i) {
            cur -> next = new Node(i);
            cur = cur -> next;
        }
    }

    int get() {
        if (!head -> next) return -1;
        Node* node = head -> next;
        head -> next = node -> next;
        isSlotAvailable[node -> val] = false;
        int val = node -> val;
        delete node;
        return val;
    }

    bool check(int number) {
        return isSlotAvailable[number];
    }

    void release(int number) {
        if (isSlotAvailable[number]) return;
        Node* node = new Node(number);
        node -> next = head -> next;
        head -> next = node;
        isSlotAvailable[number] = true;
    }
};

/**
 * Your PhoneDirectory object will be instantiated and called as such:
 * PhoneDirectory* obj = new PhoneDirectory(maxNumbers);
 * int param_1 = obj -> get();
 * bool param_2 = obj -> check(number);
 * obj -> release(number);
 */