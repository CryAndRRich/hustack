class ZigzagIterator {
    vector<int>& v1;
    vector<int>& v2;
    int i1 = 0, i2 = 0;
    int turn = 0;

public:
    ZigzagIterator(vector<int>& v1, vector<int>& v2) : v1(v1), v2(v2) {}

    int next() {
        for (int _ = 0; _ < 2; ++_) {
            if (turn % 2 == 0 && i1 < v1.size()) {
                turn++;
                return v1[i1++];
            } else if (turn % 2 == 1 && i2 < v2.size()) {
                turn++;
                return v2[i2++];
            }
            turn++;
        }
        return -1;
    }

    bool hasNext() {
        return i1 < v1.size() || i2 < v2.size();
    }
};


/**
 * Your ZigzagIterator object will be instantiated and called as such:
 * ZigzagIterator i(v1, v2);
 * while (i.hasNext()) cout << i.next();
 */