class AutocompleteSystem {
    struct TrieNode {
        unordered_map<char, TrieNode*> children;
        unordered_map<string, int> counts;
    };

    TrieNode* root;
    TrieNode* currNode;
    string currInput;

public:
    AutocompleteSystem(vector<string>& sentences, vector<int>& times) {
        root = new TrieNode();
        currNode = root;
        currInput = "";
        for (int i = 0; i < sentences.size(); ++i) {
            insert(sentences[i], times[i]);
        }
    }

    void insert(string sentence, int time) {
        TrieNode* node = root;
        for (char c : sentence) {
            if (!node->children[c])
                node->children[c] = new TrieNode();
            node = node->children[c];
            node->counts[sentence] += time;
        }
    }

    vector<string> input(char c) {
        if (c == '#') {
            insert(currInput, 1);
            currInput = "";
            currNode = root;
            return {};
        }

        currInput += c;
        if (currNode && currNode->children.count(c))
            currNode = currNode->children[c];
        else {
            currNode = nullptr;
            return {};
        }

        auto cmp = [](pair<string, int>& a, pair<string, int>& b) {
            return a.second == b.second ? a.first > b.first : a.second < b.second;
        };
        priority_queue<pair<string, int>, vector<pair<string, int>>, decltype(cmp)> pq(cmp);

        for (auto& p : currNode->counts)
            pq.push({p.first, p.second});

        vector<string> res;
        for (int i = 0; i < 3 && !pq.empty(); ++i) {
            res.push_back(pq.top().first);
            pq.pop();
        }
        return res;
    }
};

/**
 * Your AutocompleteSystem object will be instantiated and called as such:
 * AutocompleteSystem* obj = new AutocompleteSystem(sentences, times);
 * vector<string> param_1 = obj->input(c);
 */