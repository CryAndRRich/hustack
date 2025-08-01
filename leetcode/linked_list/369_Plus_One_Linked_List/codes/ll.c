struct ListNode {
    int val;
    struct ListNode* next;
};

int helper(struct ListNode* node) {
    if (!node) return 1;
    int carry = helper(node -> next);
    node -> val += carry;
    if (node -> val == 10) {
        node -> val = 0;
        return 1;
    }
    return 0;
}

struct ListNode* plusOne(struct ListNode* head) {
    if (helper(head)) {
        struct ListNode* newHead = malloc(sizeof(struct ListNode));
        newHead -> val = 1;
        newHead -> next = head;
        return newHead;
    }
    return head;
}
