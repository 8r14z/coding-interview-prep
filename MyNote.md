1. Read the problem carefully, every words in the problem statement could be a requirement/hint that needs to be covered
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
5. 