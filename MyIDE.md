# This is the IDE I mainly use to practice 

t = int(input())
n = int(input())

c = 0
tmp = n
while tmp:
    tmp &= tmp-1
    c += 1

print(n-c)