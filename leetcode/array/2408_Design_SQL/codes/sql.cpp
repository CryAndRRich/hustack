#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

using namespace std;

class SQL {
    struct Table {
        int cols;
        unordered_map<int, vector<string>> rows;
        int nextId = 1;
    };

    unordered_map<string, Table> tables;

public:
    SQL(vector<string>& names, vector<int>& columns) {
        for (int i = 0; i < names.size(); ++i)
            tables[names[i]] = Table{columns[i]};
    }

    bool ins(string name, vector<string> row) {
        if (!tables.count(name)) return false;
        Table& t = tables[name];
        if (row.size() != t.cols) return false;
        t.rows[t.nextId++] = row;
        return true;
    }

    void rmv(string name, int rowId) {
        if (tables.count(name))
            tables[name].rows.erase(rowId);
    }

    string sel(string name, int rowId, int columnId) {
        if (!tables.count(name)) return "<null>";
        Table& t = tables[name];
        if (!t.rows.count(rowId) || columnId < 1 || columnId > t.cols)
            return "<null>";
        return t.rows[rowId][columnId - 1];
    }

    vector<string> exp(string name) {
        vector<string> result;
        if (!tables.count(name)) return result;
        Table& t = tables[name];
        map<int, vector<string>> sorted(t.rows.begin(), t.rows.end());
        for (auto& [id, row] : sorted) {
            string line = to_string(id);
            for (const string& val : row)
                line += "," + val;
            result.push_back(line);
        }
        return result;
    }
};


/**
 * Your SQL object will be instantiated and called as such:
 * SQL* obj = new SQL(names, columns);
 * bool param_1 = obj->ins(name,row);
 * obj->rmv(name,rowId);
 * string param_3 = obj->sel(name,rowId,columnId);
 * vector<string> param_4 = obj->exp(name);
 */