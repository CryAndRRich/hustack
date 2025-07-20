#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>

using namespace std;
string stripLeadingZeros(const string &str) {
    size_t pos = 0;
    while (pos < str.size() - 1 && str[pos] == '0') {
        pos++;
    }
    return str.substr(pos);
}

string bigAdd(const string &num1, const string &num2) {
    int len1 = num1.size();
    int len2 = num2.size();
    int carry = 0;
    string result;

    int i = len1 - 1, j = len2 - 1;
    while (i >= 0 || j >= 0 || carry) {
        int digit1 = (i >= 0) ? num1[i] - '0' : 0;
        int digit2 = (j >= 0) ? num2[j] - '0' : 0;
        int sum = digit1 + digit2 + carry;
        carry = sum / 10;
        result.push_back((sum % 10) + '0');
        i--;
        j--;
    }
    reverse(result.begin(), result.end());
    return result;
}

string bigSubtract(const string &num1, const string &num2) {
    int len1 = num1.size();
    int len2 = num2.size();
    int carry = 0;
    string result(len1, '0');

    int i = len1 - 1, j = len2 - 1;
    while (i >= 0) {
        int digit1 = num1[i] - '0';
        int digit2 = (j >= 0) ? num2[j] - '0' : 0;
        int sub = digit1 - digit2 - carry;
        if (sub < 0) {
            sub += 10;
            carry = 1;
        } else {
            carry = 0;
        }
        result[i] = sub + '0';
        i--;
        j--;
    }
    return stripLeadingZeros(result);
}

string multiplyByPowerOf10(const string &num, int power) {
    string result = num;
    result.append(power, '0');
    return result;
}

string multiplySingleDigits(const string &a, const string &b) {
    int digit1 = a[0] - '0';
    int digit2 = b[0] - '0';
    int prod = digit1 * digit2;
    return to_string(prod);
}

string karatsuba(const string &x, const string &y) {
    string xStripped = stripLeadingZeros(x);
    string yStripped = stripLeadingZeros(y);
    int lenX = xStripped.size();
    int lenY = yStripped.size();

    if (lenX == 1 && lenY == 1) {
        return multiplySingleDigits(xStripped, yStripped);
    }

    int n = max(lenX, lenY);
    if (n % 2 != 0) {
        n++;  
    }

    string X(n - lenX, '0');
    X += xStripped;
    string Y(n - lenY, '0');
    Y += yStripped;

    int m = n / 2;
    string X_high = X.substr(0, n - m);
    string X_low  = X.substr(n - m, m);
    string Y_high = Y.substr(0, n - m);
    string Y_low  = Y.substr(n - m, m);

    string z2 = karatsuba(X_high, Y_high);
    string z0 = karatsuba(X_low, Y_low);

    string sum1 = bigAdd(X_high, X_low);
    string sum2 = bigAdd(Y_high, Y_low);
    string z1 = karatsuba(sum1, sum2);

    string temp = bigAdd(z2, z0);
    string z1_final = bigSubtract(z1, temp);

    string z2_shift = multiplyByPowerOf10(z2, 2 * m);
    string z1_shift = multiplyByPowerOf10(z1_final, m);
    string temp2 = bigAdd(z2_shift, z1_shift);
    string result = bigAdd(temp2, z0);

    return stripLeadingZeros(result);
}

int main() {
    string a, b;
    cin >> a >> b;
    
    string result = karatsuba(a, b);
    cout << result << endl;
    return 0;
}
