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