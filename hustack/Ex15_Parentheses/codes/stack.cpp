#include <iostream>
#include <stack>
#include <string>
using namespace std;

bool check_correct(const string &s) {
    stack<char> stk;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            stk.push(c);
        } else {
            if (stk.empty()) return false;
            char top = stk.top(); stk.pop();
            if ((c == ')' && top != '(') ||
                (c == ']' && top != '[') ||
                (c == '}' && top != '{')) {
                return false;
            }
        }
    }
    return stk.empty();
}

int main() {
    string s;
    cin >> s;
    cout << (check_correct(s) ? 1 : 0) << endl;
    return 0;
}
