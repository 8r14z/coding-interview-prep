# t = int(input())
# for _ in range(t):
#     n = int(input())
#     msb = 0
	

tmp = int(input())
msb = 0
while tmp:
	tmp = tmp>>1
	msb += 1
print(2**(msb-1) - 1)