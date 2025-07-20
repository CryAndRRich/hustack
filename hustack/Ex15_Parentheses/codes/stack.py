def check_correct(s: str) -> int:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in s:
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                return 0
            stack.pop()
    return 1 if not stack else 0

if __name__ == "__main__":
    s = input().strip()
    print(check_correct(s))
