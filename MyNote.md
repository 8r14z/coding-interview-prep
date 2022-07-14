# DISCLAIMER
This note only makes sense to me :yaoming: 

# Notes that i learned by heart while solving problems
1. Read the problem carefully, every words in the problem statement could be a requirement/hint that needs to be covered
IMPORTANT: read the input/output carefully, and think of different cases, dont assume that the input is unique and doesnt have any variant. For example: one integer/type can appear multiple times in an array

Read input/output/constraint...

2. Sometimes Binary Search Tree is hard to implement, Stack+Sort can come to rescue 

3. Python List is not hashable bro! Remember :'(

4. Rotate matrix 90 degree
```python
# Extra space
for i in range(m):
    for j in range(n):
        newMatrix[i][j] = matrix[n-1-j][i]

for i in range(n):
    for j in range(m):
        newMatrix[j][n-1-i] = matrix[i][j]

# In-place with nxn matrix
n = len(matrix)        
for i in range(n):
    for j in range(i, n):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
for i in range(n):
    for j in range(n//2):
        matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
```

5. 2 pointer problems.. 2 pointers usually start at the same position, one move faster than the other

6. the prolem statement usually has some hint about input validation. for example: input is array of 0 and 1 and number of 0 and 1 are equal, output is array of alternating binary 0101... or 1010... So if there is one char (0 or 1) is not in correct possition so there is another (1 or 0) is either. meaning number of invalid positions are equal. it never has many invalid positions in one number and less invalid positions in the other cuz it will cause the imbalance in number of 0 and 1

7. Repeated numnbers can be represented in a hash table with frequencies 
```python
hashNums1 = self.makeHashCount(nums1)
hashNums2 = self.makeHashCount(nums2)

for num1 in hashNums1:
    if target-num1 in hashNums2:
        count += (hashNums1[num1] * hashNums2[target-num1])
```

8. Number of subsets in a list with N elements = 2<sup>N</sup>. if there is N numbers, there are 2<sup>N</sup>-1 reperesentation of their combination in bit mask. For example:

arr=[1,2,3]

Bit mask:
|Binary|Decimal|
|-|-|
|0001|1|
|0010|2|
|0011|3|
|0100|4|
|0101|5|
|0110|6|
|0111|7|
...|...
```python
for mask in range(2**N):
    sum = 0
    for i in range(N): 
        # at this step, it shifts index to match with the mask, so depends on the mask it will get combination of 1 or 2 or 3 elements together
        # each mask is used to filter out the combination of numbers
        # for example 0011 will be the combination of [a[0], a[1]]
        # 0100 will be the subset of [a[2]]
        # 0111 is subset of [a[0], a[1], a[2]]
        # and so on..
        if mask & (1 << i) > 0:
           sum += arr[i] 
```

9. DP pattern can be recognized when the current problem result can be influenced by the smaller sub-problem :) The DP problem usually is stated by "find number of works ....", "find min number works", "find max number of works", "longest something", "shortest something",...

10. Sometime we can use the algorithm to find the smallest to find the largest (or opposite) by reverting all numbers to opposite sign (1 -> -1, -1 -> 1, etc). \
This trick is applied when u wanna use heapq implementation in Python, by default it's min-heap, if we wawnna covert it to max-heap, can just multiply all elements by -1 and use the min-heap :) \
Another example for this trick is to find the longest path, we know how to find the shortest path with Dijkstra and Bellman-Ford, so we can just revert the sign of weight and find the longest path with Bellman-Ford as Dijstra doesnt work for negative weight :D 

11. Find k with n&(n-1)&(n-2)&...k = 0 \

To make it 0 with & at the end, we need to have 0 at all position in bit presentation 0x000000.
```
01|11   7
01|10   6
01|01   5
01|00   4
00|11   3
 ^
```
We can see in decreasing order, MSB will be keep the same as 1 (from 7 -> 4) until it meet new presentation starts with 0 as 0x0111...1. So this is the max number can make the whole serie & = 0. To get this number, we simply can get 4 - 1 in the example. Generally, get MSB left of `n`, let's call i, so 2<sup>i</sup> - 1 is `k` 

```python
n = int(input())
print(2**(len(bin(n))-3) - 1)
# bin here is to get the bit presetantion of an number
# 5 => '0b101'
# so to get most significant bit, len(0b101) - 3 = 2 => k = 2**2 -1 = 3
# can use >> in this case also to find the msb

msb = 0
while n:
	n >>= 1
	msb += 1
print(2**(msb-1) - 1)
```

12. Travel half of array, i in range((n+1)//2) for both odd and even
- even: n = 4 => i in range(3) => i in {0,1,2}
```
0 1 2 3
    ^
```
- odd: n = 7 => i in range(4) => i in {0,1,2,3}
```
0 1 2 3 4
      ^
```
Travel the same number of items left and right from mid, i in range(n//2)
- even: n = 4 -> i in range(2) => left in {0,1}  right in {2,3}
```
0 1 | 2 3
  L   R 
```
- odd: n = 5 -> i in range(2) => left in {0,1} right in {3,4}
```
0 1 2 3 4
  L | R 
```

```python
for i in range(n//2):
    if a[n//2-1-i] == a[(n+1)//2+i]:
        print(a[i])
```

13. Fibonacci with DP is O(n) time - O(n) space, optimize by saving 2 running variables instead of array -> O(n) time - O(1) space

Can improve fib to O(logN) time: https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/

14. HEAP HEAP HEAP HEAP

15. Sometimes bruce-force is good enough

16. Shortest path algorithms: 
- Unweighted: BFS
- Weighted: 
    - no cycle: DAG
    - no negative weight and no negative cycle: Dijkstra
    - negative weight or negative cycle: Bellmen-Ford 

17. Divide and conquer
S1. Split problem in to smaller subproblems
S2. Continously split the subproblems into smaller subproblems till a subproblem is small enough
S3. Recursively solve the subproblem
S4. Merge 2 subproblems

18. Python bisect.bisect => left <= i, right > i .... bisect.bisect_left => left < i, right >= i

19. If you can't figure out a dynamic programming solution, you can always do DFS + memoization which does the same thing.