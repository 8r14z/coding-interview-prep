# A - And Then There Were K

t = int(input())
for _ in range(t):
    n = int(input())
    msb = 0
    while n:
	    n >>= 1
	    msb += 1
    print(2**(msb-1) - 1)
