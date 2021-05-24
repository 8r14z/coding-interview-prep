# This is the IDE I mainly use to practice

def solve(x):
    arr = [int(i) for i in input().split()]
    n = arr[0]
    k = arr[1]
    S = str(input())

    res = 0

    for i in range((n+1)//2):
        res += (ord(S[i]) - 97)*(k**(mid-i-1))

    isSmaller = False
    for i in range(n//2):
        if S[-i-1] < S[mid+i]:
            isSmaller = True

    if isSmaller:
        res += 1

        print(f'Case #{x+1}: {res}')


t = int(input())

for x in range(t):
    solve(x)
