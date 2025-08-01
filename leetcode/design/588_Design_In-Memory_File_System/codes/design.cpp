class FileSystem {
    struct Node {
        bool isFile = false;
        string content;
        map<string, Node*> children;
    };
    
    Node* root;

public:
    FileSystem() {
        root = new Node();
    }

    vector<string> ls(string path) {
        Node* node = traverse(path);
        if (node -> isFile) return {path.substr(path.find_last_of('/') + 1)};
        vector<string> res;
        for (auto& p : node -> children) res.push_back(p.first);
        return res;
    }

    void mkdir(string path) {
        traverse(path, true);
    }

    void addContentToFile(string filePath, string content) {
        Node* node = traverse(filePath, true);
        node -> isFile = true;
        node -> content += content;
    }

    string readContentFromFile(string filePath) {
        Node* node = traverse(filePath);
        return node -> content;
    }

private:
    Node* traverse(const string& path, bool create = false) {
        Node* node = root;
        stringstream ss(path);
        string part;
        getline(ss, part, '/'); 
        while (getline(ss, part, '/')) {
            if (!node -> children.count(part))
                node -> children[part] = new Node();
            node = node -> children[part];
        }
        return node;
    }
};

/**
 * Your FileSystem object will be instantiated and called as such:
 * FileSystem* obj = new FileSystem();
 * vector<string> param_1 = obj -> ls(path);
 * obj -> mkdir(path);
 * obj -> addContentToFile(filePath,content);
 * string param_4 = obj -> readContentFromFile(filePath);
 */