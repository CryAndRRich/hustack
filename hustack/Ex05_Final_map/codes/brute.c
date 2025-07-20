#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CARS 50000

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    
    double xs[MAX_CARS], ys[MAX_CARS];
    
    for (int i = 0; i < n; i++) {
        scanf("%lf %lf", &xs[i], &ys[i]);
    }
    for (int q = 0; q < m; q++) {
        double left, top, right, bottom;
        scanf("%lf %lf %lf %lf", &left, &top, &right, &bottom);
        
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (xs[i] > left && xs[i] < right && ys[i] < top && ys[i] > bottom)
                count++;
        }
        printf("%d\n", count);
    }
    
    return 0;
}
