#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 10010

typedef struct {
    char ch;
    int freq;
} CharFreq;

int cmp(const void* a, const void* b) {
    CharFreq* x = (CharFreq*)a;
    CharFreq* y = (CharFreq*)b;
    return y -> freq - x -> freq;
}

char* rearrangeString(char* s, int k) {
    if (k == 0) return strdup(s);

    int freq[26] = {0}, len = strlen(s);
    for (int i = 0; s[i]; i++) freq[s[i] - 'a']++;

    int mf = 0;
    for (int i = 0; i < 26; i++)
        if (freq[i] > mf) mf = freq[i];

    char head[27] = "";
    for (int i = 0; i < 26; i++)
        if (freq[i] == mf) {
            int l = strlen(head);
            head[l] = 'a' + i;
            head[l+1] = '\0';
        }

    if (k * (mf - 1) + strlen(head) > len)
        return strdup("");

    char** res = malloc(mf * sizeof(char*));
    for (int i = 0; i < mf; i++) {
        res[i] = malloc(MAX_LEN);
        strcpy(res[i], head);
    }

    CharFreq others[26];
    int idx = 0;
    for (int i = 0; i < 26; i++) {
        if (freq[i] != mf && freq[i] > 0) {
            others[idx].ch = 'a' + i;
            others[idx].freq = freq[i];
            idx++;
        }
    }

    qsort(others, idx, sizeof(CharFreq), cmp);

    int j = 0;
    for (int i = 0; i < idx; i++) {
        for (int c = 0; c < others[i].freq; c++) {
            strcat(res[j], (char[]){others[i].ch, '\0'});
            j = (j + 1) % (mf - 1);
        }
    }

    char* ans = malloc(MAX_LEN);
    ans[0] = '\0';
    for (int i = 0; i < mf; i++) {
        strcat(ans, res[i]);
        free(res[i]);
    }
    free(res);

    return ans;
}
