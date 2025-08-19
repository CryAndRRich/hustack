#include <stdlib.h>

/**
Definition for singly-linked list.
 */
struct ListNode {
    int val;
    struct ListNode *next;
};

struct ListNode* deleteDuplicatesUnsorted(struct ListNode* head) {
    int* count = (int*)calloc(100001, sizeof(int));
    struct ListNode* curr = head;
    while (curr) {
        count[curr -> val]++;
        curr = curr -> next;
    }
    struct ListNode dummy;
    dummy.next = head;
    struct ListNode* prev = &dummy;
    curr = head;
    while (curr) {
        if (count[curr -> val] > 1) {
            prev -> next = curr -> next;
        } else {
            prev = curr;
        }
        curr = curr -> next;
    }
    free(count);
    return dummy.next;
}
