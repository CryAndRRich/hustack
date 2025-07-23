using namespace std;
class Solution {
public:
    int calculate(string s) {
        int i = 0;
        return helper(s, i);
    }

private:
    int helper(const string& s, int& i) {
        stack<int> stk;
        int num = 0;
        char sign = '+';

        while (i < s.length()) {
            char c = s[i++];

            if (isdigit(c)) {
                num = num * 10 + (c - '0');
            }

            if (c == '(') {
                num = helper(s, i);
            }

            if ((!isdigit(c) && c != ' ') || i == s.length()) {
                if (sign == '+') stk.push(num);
                else if (sign == '-') stk.push(-num);
                else if (sign == '*') {
                    int temp = stk.top(); stk.pop();
                    stk.push(temp * num);
                }
                else if (sign == '/') {
                    int temp = stk.top(); stk.pop();
                    stk.push(temp / num);
                }
                sign = c;
                num = 0;
            }

            if (c == ')') break;
        }

        int sum = 0;
        while (!stk.empty()) {
            sum += stk.top(); stk.pop();
        }
        return sum;
    }
};
