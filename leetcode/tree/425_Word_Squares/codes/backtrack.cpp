#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int N;
    unordered_map<string, vector<string>> prefixHashTable;
    vector<string> words;

    vector<vector<string>> wordSquares(vector<string>& words) {
        this -> words = words;
        this -> N = words[0].size();
        buildPrefixHashTable(words);

        vector<vector<string>> results;
        for (string& word : words) {
            vector<string> wordSquare{word};
            backtracking(1, wordSquare, results);
        }
        return results;
    }

    void backtracking(int step, vector<string>& wordSquare, vector<vector<string>>& results) {
        if (step == N) {
            results.push_back(wordSquare);
            return;
        }

        string prefix;
        for (int i = 0; i < step; ++i) {
            prefix += wordSquare[i][step];
        }

        for (string& candidate : getWordsWithPrefix(prefix)) {
            wordSquare.push_back(candidate);
            backtracking(step + 1, wordSquare, results);
            wordSquare.pop_back();
        }
    }

    void buildPrefixHashTable(vector<string>& words) {
        for (string& word : words) {
            for (int i = 1; i < N; ++i) {
                string prefix = word.substr(0, i);
                prefixHashTable[prefix].push_back(word);
            }
        }
    }

    vector<string> getWordsWithPrefix(const string& prefix) {
        if (prefixHashTable.count(prefix))
            return prefixHashTable[prefix];
        return {};
    }
};