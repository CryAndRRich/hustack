def check_correct(s):
    prev = None
    while s != prev:
        prev = s
        s = s.replace("()", "").replace("[]", "").replace("{}", "")
    return 1 if s == "" else 0

if __name__ == "__main__":
    s = input().strip()
    print(check_correct(s))