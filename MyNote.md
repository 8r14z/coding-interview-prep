# DISCLAIMER:
This note might only make sense to me :yaoming: 
--------
1. Read the problem carefully, every words in the problem statement could be a requirement/hint that needs to be covered
IMPORTANT: read the input/output carefully, and think of different cases, dont assume that the input is unique and doesnt have any variant. For example: one integer/type can appear multiple times in an array

Read input/output/constraint...

2. Sometimes Binary Search Tree is hard to implement, Stack+Sort can come to rescue 

3. Python List is not hashable bro! Remember :'(

4. Rotate matrix 90 degree
```python
for i in range(m):
    for j in range(n):
        newMatrix[i][j] = matrix[n-1-j][i]

for i in range(n):
    for j in range(m):
        newMatrix[j][n-1-i] = matrix[i][j]
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