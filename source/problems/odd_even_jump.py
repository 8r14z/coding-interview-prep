# https://leetcode.com/problems/odd-even-jump/
class Solution:
    def oddEvenJumps(self, A):
        n = len(A)
        next_larger = self.next_jumps(sorted(enumerate(A), key=lambda tup: tup[1]))
        next_smaller = self.next_jumps(sorted(enumerate(A), key=lambda tup: tup[1], reverse=True))

        oddJumps = [False] * n
        evenJumps = [False] * n
        oddJumps[-1] = evenJumps[-1] = True

        for i in range(n-2, -1, -1):
            oddJumps[i] = evenJumps[next_larger[i]]
            evenJumps[i] = oddJumps[next_smaller[i]]

        return sum(oddJumps)

    def next_jumps(self, sorted_indices):
        stack = []
        output = [0] * len(sorted_indices)
        for i, val in sorted_indices:
            while stack and i > stack[-1]:
                top = stack.pop()
                output[top] = i
            stack.append(i)
        return output