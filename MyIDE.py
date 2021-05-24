def solve(s, left, right, pleft, pright, k, n):

	pleft = left-1
	pright = right+1

	if pleft is not None and pright is not None and pleft < :
		

	res = 0
	
	return res

t = int(input())

for x in range(t):
	arr = [int(i) for i in input().split()]
	N = arr[0]
	K = arr[1]
	S = str(input())

	res = 0
	if N == 1:
		res = ord(S[0]) - 97
	else:
		res = solve(S, 0, N-1, None, None, K, N)

	print(f'Case #{x+1}: {res}')	