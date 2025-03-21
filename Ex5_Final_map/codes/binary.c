#include <stdio.h>
#include <stdlib.h>

#define MAX 50000

double xs[MAX], ys[MAX];
int idx[MAX];

int cmp(const void *a, const void *b) {
    int ia = *(int *)a;
    int ib = *(int *)b;
    if (xs[ia] < xs[ib])
        return -1;
    else if (xs[ia] > xs[ib])
        return 1;
    else
        return 0;
}

int lower_bound_strict(int n, double key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (xs[idx[mid]] > key)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

int lower_bound_nonstrict(int n, double key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (xs[idx[mid]] >= key)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

int main(){
    int n, m;
    scanf("%d %d", &n, &m);
    
    for (int i = 0; i < n; i++){
        scanf("%lf %lf", &xs[i], &ys[i]);
        idx[i] = i;
    }
    
    qsort(idx, n, sizeof(int), cmp);
    
    for (int q = 0; q < m; q++){
        double left, top, right, bottom;
        scanf("%lf %lf %lf %lf", &left, &top, &right, &bottom);
        
        int lo = lower_bound_strict(n, left);
        int hi = lower_bound_nonstrict(n, right);
        
        int count = 0;
        for (int i = lo; i < hi; i++){
            int pos = idx[i];
            if (ys[pos] > bottom && ys[pos] < top)
                count++;
        }
        printf("%d\n", count);
    }
    
    return 0;
}
