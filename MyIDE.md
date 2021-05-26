# This is the IDE I mainly use to practice

def solve(x):
    arr = [int(i) for i in input().split()]
    n = arr[0]
    k = arr[1]
    S = str(input())

    res = 0
    mod = 10**9 + 7
    mid = (n+1)//2
    for i in range(mid):
        res += (ord(S[i]) - 97)* pow(k, mid-i-1, mod)

    isSmaller = False
    for i in range(n//2):
        if S[n//2-1-i] == S[mid+i]:
            continue
        if S[n//2-1-i] < S[mid+i]:
            isSmaller = True
          break

    if isSmaller:
        res = (res+1)
        
    print(f'Case #{x+1}: {int(res%mod)}')


t = int(input())

for x in range(t):
    solve(x)
