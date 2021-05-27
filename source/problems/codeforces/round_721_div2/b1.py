# B1 - Palindrome Game (easy version)
# https://codeforces.com/contest/1527/problem/B1
def solve(x):
    n = int(input())
    s = str(input())

    numOfZeros = 0
    for c in s:
        if c == '0':
            numOfZeros += 1

    if numOfZeros%2 == 0:
        print('BOB')
    else:
        if numOfZeros == 1:
            print('BOB')
        else:
            print('ALICE')
        
t = int(input())

for x in range(t):
    solve(x)

# ALICE starts first
# BOB's strategy is to not allow ALICE has chance to reverse string
# if number of zeros is even, whenever ALICE make 0 -> 1 at i, BOB trying to make it palindrom in his turn by make 0 -> 1 at n-i-1
# At the end, 1 last 0 left, 1101, BOB reverse to make ALICE do the last one. -> BOB wins
# in case the num of zeros is odd, ALICE starts with make 0 -> 1 at i = n/2 and apply same strategy as BOB 