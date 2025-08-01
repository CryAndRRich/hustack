#include <unordered_map>
#include <vector>
#include <algorithm>

class Leaderboard {
private:
    std::unordered_map<int, int> scores;

public:
    Leaderboard() {}

    void addScore(int playerId, int score) {
        scores[playerId] += score;
    }

    int top(int K) {
        std::vector<int> vals;
        for (const auto& pair : scores)
            vals.push_back(pair.second);
        std::sort(vals.rbegin(), vals.rend());
        int sum = 0;
        for (int i = 0; i < K && i < vals.size(); ++i)
            sum += vals[i];
        return sum;
    }

    void reset(int playerId) {
        scores[playerId] = 0;
    }
};

/**
 * Your Leaderboard object will be instantiated and called as such:
 * Leaderboard* obj = new Leaderboard();
 * obj->addScore(playerId,score);
 * int param_2 = obj->top(K);
 * obj->reset(playerId);
 */