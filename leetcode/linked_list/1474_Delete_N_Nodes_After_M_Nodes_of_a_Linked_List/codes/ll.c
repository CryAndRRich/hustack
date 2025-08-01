/**
 * Definition for singly-linked list.
 */

struct ListNode {
    int val;
    struct ListNode *next;
};
 
struct ListNode* deleteNodes(struct ListNode* head, int m, int n) {
    struct ListNode* curr = head;
    while (curr != NULL) {
        for (int i = 1; i < m && curr != NULL; i++) {
            curr = curr -> next;
        }
        if (curr == NULL) break;

        struct ListNode* temp = curr -> next;
        for (int i = 0; i < n && temp != NULL; i++) {
            struct ListNode* toDelete = temp;
            temp = temp -> next;
        }
        curr -> next = temp;
        curr = temp;
    }
    return head;
}