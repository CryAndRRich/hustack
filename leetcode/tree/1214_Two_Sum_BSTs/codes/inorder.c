#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

void inorder(struct TreeNode* root, int* arr, int* idx) {
    if (!root) return;
    inorder(root -> left, arr, idx);
    arr[(*idx)++] = root -> val;
    inorder(root -> right, arr, idx);
}

bool twoSumBSTs(struct TreeNode* root1, struct TreeNode* root2, int target) {
    int arr1[5000], arr2[5000], n1 = 0, n2 = 0;
    inorder(root1, arr1, &n1);
    inorder(root2, arr2, &n2);

    int i = 0, j = n2 - 1;
    while (i < n1 && j >= 0) {
        int sum = arr1[i] + arr2[j];
        if (sum == target) return true;
        else if (sum < target) i++;
        else j--;
    }
    return false;
}
