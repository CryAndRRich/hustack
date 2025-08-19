#include <unordered_map>
using namespace std;

/**
Definition for singly-linked list.
 */
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* deleteDuplicatesUnsorted(ListNode* head) {
        unordered_map<int, int> count;
        ListNode* curr = head;
        while (curr) {
            count[curr -> val]++;
            curr = curr -> next;
        }
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prev = &dummy;
        curr = head;
        while (curr) {
            if (count[curr -> val] > 1) {
                prev -> next = curr -> next;
            } else {
                prev = curr;
            }
            curr = curr -> next;
        }
        return dummy.next;
    }
};
