# A - And Then There Were K
# https://codeforces.com/contest/1527/problem/A
t = int(input())
for _ in range(t):
    n = int(input())
    msb = 0
    while n:
	    n >>= 1
	    msb += 1
    print(2**(msb-1) - 1)
