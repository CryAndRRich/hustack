#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int key;
    struct Node *next;
} Node;

Node* createNode(int key) {
    Node* newNode = (Node*)malloc(sizeof(Node));

    newNode -> key = key;
    newNode -> next = NULL;
    return newNode;
}

Node* findNode(Node* head, int key) {
    Node* temp = head;
    while(temp != NULL) {
        if (temp -> key == key)
            return temp;

        temp = temp -> next;
    }
    return NULL;
}

void addFirst(Node** head, int key) {
    if (findNode(*head, key) != NULL)
        return; 
    Node* newNode = createNode(key);
    newNode -> next = *head;
    *head = newNode;
}

void addLast(Node** head, int key) {
    if (findNode(*head, key) != NULL)
        return;

    Node* newNode = createNode(key);
    if (*head == NULL) {
        *head = newNode;
    } else {
        Node* temp = *head;
        while(temp -> next != NULL)
            temp = temp -> next;
        temp -> next = newNode;
    }
}

void addAfter(Node** head, int newKey, int afterKey) {
    if (findNode(*head, newKey) != NULL)
        return;

    Node* afterNode = findNode(*head, afterKey);
    if (afterNode == NULL)
        return;

    Node* newNode = createNode(newKey);
    newNode -> next = afterNode -> next;
    afterNode -> next = newNode;
}

void addBefore(Node** head, int newKey, int beforeKey) {
    if (findNode(*head, newKey) != NULL)
        return;

    if (*head == NULL)
        return;
        
    if ((*head) -> key == beforeKey) {
        addFirst(head, newKey);
        return;
    }

    Node* prev = *head;
    Node* cur = (*head) -> next;
    while(cur != NULL && cur -> key != beforeKey) {
        prev = cur;
        cur = cur -> next;
    }

    if (cur == NULL)
        return;

    Node* newNode = createNode(newKey);
    newNode -> next = cur;
    prev -> next = newNode;
}

void removeNode(Node** head, int key) {
    if (*head == NULL)
        return;

    Node* temp = *head;
    
    if (temp -> key == key) {
        *head = temp -> next;
        free(temp);
        return;
    }

    Node* prev = temp;
    temp = temp -> next;
    while(temp != NULL && temp -> key != key) {
        prev = temp;
        temp = temp -> next;
    }

    if (temp == NULL)
        return; 

    prev -> next = temp -> next;
    free(temp);
}

void reverseList(Node** head) {
    Node* prev = NULL;
    Node* curr = *head;
    Node* next = NULL;
    while(curr != NULL) {
        next = curr -> next;
        curr -> next = prev;
        prev = curr;
        curr = next;
    }
    *head = prev;
}

int main(){
    int n;
    scanf("%d", &n);
    
    Node* head = NULL;
    for (int i = 0; i < n; i++){
        int key;
        scanf("%d", &key);
        addLast(&head, key);
    }
    
    char command[50];
    while(1){
        scanf("%s", command);
        if (command[0] == '#' && command[1] == '\0'){
            break;
        }
        if (strcmp(command, "addlast") == 0){
            int k;
            scanf("%d", &k);
            addLast(&head, k);
        }
        else if (strcmp(command, "addfirst") == 0){
            int k;
            scanf("%d", &k);
            addFirst(&head, k);
        }
        else if (strcmp(command, "addafter") == 0){
            int u, v;
            scanf("%d %d", &u, &v);
            addAfter(&head, u, v);
        }
        else if (strcmp(command, "addbefore") == 0){
            int u, v;
            scanf("%d %d", &u, &v);
            addBefore(&head, u, v);
        }
        else if (strcmp(command, "remove") == 0){
            int k;
            scanf("%d", &k);
            removeNode(&head, k);
        }
        else if (strcmp(command, "reverse") == 0){
            reverseList(&head);
        }
    }
    
    Node* curr = head;
    while(curr != NULL){
        printf("%d", curr -> key);
        if (curr -> next != NULL)
            printf(" ");
        curr = curr -> next;
    }
    
    return 0;
}
