class Solution:
    def calculate(self, s: str) -> int:
        def helper(chars: list) -> int:
            stack = []
            num = 0
            sign = '+'

            while chars:
                c = chars.pop(0)

                if c.isdigit():
                    num = num * 10 + int(c)

                if c == '(':
                    num = helper(chars)

                if (not c.isdigit() and c != ' ') or not chars:
                    if sign == '+':
                        stack.append(num)
                    elif sign == '-':
                        stack.append(-num)
                    elif sign == '*':
                        stack[-1] *= num
                    elif sign == '/':
                        stack[-1] = int(stack[-1] / num)
                    sign = c
                    num = 0

                if c == ')':
                    break

            return sum(stack)

        return helper(list(s))
