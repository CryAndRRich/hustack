class Solution {
public:
    bool verifyPreorder(vector<int>& preorder) {
        int low = INT_MIN;
        int i = -1;
        for (int val : preorder) {
            if (val < low)
                return false;

            while (i >= 0 && val > preorder[i]) {
                low = preorder[i--];
            }

            preorder[++i] = val;
        }
        return true;
    }
};
