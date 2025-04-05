#include <iostream>
#include <bits/stdc++.h>

using namespace std;

int gcd(int a, int b) {
    if (a < b) {
        a = a ^ b;
        b = a ^ b;
        a = a ^ b;
    }
    if (a % b == 0) {
        return b;
    }
    return gcd(b, a % b);
}

int pourWater(int start, int end, int target) {
    int one = start;
    int two = 0;
    int step = 1;
    
    while(one != target && two != target) {
        int trans = min(one, end - two);
        one -= trans;
        two += trans;
        step += 1;
        
        if(one == target || two == target) {
            break;
        }
        
        if(one == 0) {
            one = start;
            step += 1;
        }
        if(two == end) {
            two = 0;
            step += 1;
        }
    }
    return step;
}

int minStep(int a, int b, int c) {
    int meth1 = pourWater(a, b, c);
    int meth2 = pourWater(b, a, c);
    return min(meth1, meth2);
}

int main()
{
    int a, b, c;
    cin >> a >> b >> c;
    if (c <= max(a, b) && c % gcd(a, b) == 0) {
        cout << minStep(a, b, c);
    } else {
        cout << -1;
    }
    return 0;
}
