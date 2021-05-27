def solve(x):
    n = int(input())
    count = 0
    while n:
        count+=1
        n = n>>1
    
    print(2**count - 1)

t = int(input())

for x in range(t):
    solve(x)
