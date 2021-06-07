# def solution(S, C):
#     n = len(S)
#     if n < 2: return 0
#     if len(C) != n: return 0

#     answer = 0
#     cur_sum = 0

#     for i in range(n-1):
#         if S[i] == S[i+1]:
#             cur_sum += C[i]
#         else:
#             answer += cur_sum
#             cur_sum = 0
    
#     answer += cur_sum

#     return answer


# print(solution('aabbcccc', [1, 2, 1, 2, 1, 2,3,4]))


def solution(S, C):
    n = len(S)
    if n < 2: return 0
    if len(C) != n: return 0

    answer = 0
    cur_sum = 0
    cur_max = 0

    tmp_s = S + '\n'

    for i in range(n):
        cur_max = max(cur_max, C[i])
        cur_sum += C[i]    

        if tmp_s[i] != tmp_s[i+1]:
            answer += (cur_sum - cur_max)
            cur_sum = 0
            cur_max = 0

    return answer


print(solution('aabbcccc', [1, 2, 1, 2, 4, 3,2,1]))

import collections

n = int(input())
columns = [cl for cl in input().split()]
Row = collections.namedtuple('Row', ','.join(columns))

sum = 0
for _ in range(n):
    cl = input().split()
    row=Row(cl[0], cl[1], cl[2], cl[3])
    sum+= int(row.MARKS)

print('{:.2f}'.format(sum/n))
