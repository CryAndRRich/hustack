#include <iostream>
#include <string>

using namespace std;

struct Node {
    int key;
    Node *next;
};

Node* createNode(int key) {
    Node* newNode = new Node;
    newNode -> key = key;
    newNode -> next = nullptr;
    return newNode;
}

Node* findNode(Node* head, int key) {
    Node* temp = head;
    while(temp != nullptr) {
        if (temp -> key == key)
            return temp;
        temp = temp -> next;
    }
    return nullptr;
}

void addFirst(Node** head, int key) {
    if (findNode(*head, key) != nullptr)
        return;
    Node* newNode = createNode(key);
    newNode -> next = *head;
    *head = newNode;
}

void addLast(Node** head, int key) {
    if (findNode(*head, key) != nullptr)
        return;
    Node* newNode = createNode(key);
    if (*head == nullptr) {
        *head = newNode;
    } else {
        Node* temp = *head;
        while(temp -> next != nullptr)
            temp = temp -> next;
        temp -> next = newNode;
    }
}

void addAfter(Node** head, int newKey, int afterKey) {
    if (findNode(*head, newKey) != nullptr)
        return;
    Node* afterNode = findNode(*head, afterKey);
    if (afterNode == nullptr)
        return;
    Node* newNode = createNode(newKey);
    newNode -> next = afterNode -> next;
    afterNode -> next = newNode;
}

void addBefore(Node** head, int newKey, int beforeKey) {
    if (findNode(*head, newKey) != nullptr)
        return;
    
    if (*head == nullptr)
        return;
        
    if ((*head) -> key == beforeKey) {
        addFirst(head, newKey);
        return;
    }
    
    Node* prev = *head;
    Node* cur = (*head) -> next;
    while(cur != nullptr && cur -> key != beforeKey) {
        prev = cur;
        cur = cur -> next;
    }
    
    if (cur == nullptr)
        return;
    
    Node* newNode = createNode(newKey);
    newNode -> next = cur;
    prev -> next = newNode;
}

void removeNode(Node** head, int key) {
    if (*head == nullptr)
        return;
    
    Node* temp = *head;
    
    if (temp -> key == key) {
        *head = temp -> next;
        delete temp;
        return;
    }
    
    Node* prev = temp;
    temp = temp -> next;
    while(temp != nullptr && temp -> key != key) {
        prev = temp;
        temp = temp -> next;
    }
    
    if (temp == nullptr)
        return;
    
    prev -> next = temp -> next;
    delete temp;
}

void reverseList(Node** head) {
    Node* prev = nullptr;
    Node* curr = *head;
    Node* next = nullptr;
    while(curr != nullptr) {
        next = curr -> next;
        curr -> next = prev;
        prev = curr;
        curr = next;
    }
    *head = prev;
}

int main() {
    int n;
    cin >> n;
    
    Node* head = nullptr;
    for (int i = 0; i < n; i++){
        int key;
        cin >> key;
        addLast(&head, key);
    }
    
    string command;
    while (true) {
        cin >> command;
        if (command == "#")
            break;
        if (command == "addlast") {
            int k;
            cin >> k;
            addLast(&head, k);
        }
        else if (command == "addfirst") {
            int k;
            cin >> k;
            addFirst(&head, k);
        }
        else if (command == "addafter") {
            int u, v;
            cin >> u >> v;
            addAfter(&head, u, v);
        }
        else if (command == "addbefore") {
            int u, v;
            cin >> u >> v;
            addBefore(&head, u, v);
        }
        else if (command == "remove") {
            int k;
            cin >> k;
            removeNode(&head, k);
        }
        else if (command == "reverse") {
            reverseList(&head);
        }
    }
    
    Node* curr = head;
    while(curr != nullptr) {
        cout << curr -> key;
        if (curr -> next != nullptr)
            cout << " ";
        curr = curr -> next;
    }
    
    while(head != nullptr) {
        Node* temp = head;
        head = head -> next;
        delete temp;
    }
    
    return 0;
}
