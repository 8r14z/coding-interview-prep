# Materials:
I. [MIT 6.006 Introduction to Algorithms](#introduction)

## [MIT 6.006 Introduction to Algorithms](https://youtube.com/playlist?list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb) <a name="introduction"></a>

https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/

```
Pro-tip: watch at 1.75 speed
```

### 1. Algorithmic Thinking, Peak Finding
Big O: The upper bound. Guarantee that the algorithm will run faster (below) than this

Big Omega: The lower bound, the algorithm is always running slower than this

Big Theta: represent both lower bound and upper bound and 2 bounds are differentiated by constant factor of the dominant factor in function

**g(x) = 1.1 x<sup>2</sup> + x<sup>1.5</sup> + 1000**

Big Theta is θ(x<sup>2</sup>) and lower bound is x<sup>2</sup>, upper bound is 1.2x<sup>2</sup> as 1.2 > 1.1 and 1.1 x<sup>2</sup> is dominant factor g(x)

In most cases, Big Theta and Big O are using interchangably, most of the time we talking about Big O in engineering industry, it's the tighest bound of the algorithm and usually it means the same thing as Big Theta in academic.

**A peak in 1-D array**
A peak is the number that larger than or equal the number in index-1 and index+1
-> Binary search, to find a peak. If mid < left, go(a[0:left]), else if mid < right, go(a[mid+1, n]), else return mid

**A peak in 2-D array**
Matrix N x M. Binary search on columns, start at m = M/2 find the max on the column m at (i,j), move left if a[i][j-1] > a[i][j], move right if a[i][j+1] > a[i][j] 

Explain: for example when moving left to column j-1, the find max on column j-1, the max is always greater than its right as it's it greater than the a[i][j-1] and a[i][j-1] > max of j

### 2. Models of Computation, Document Distance

### 3. Insertion Sort, Merge Sort
Insertion Sort can be more efficient as nlog(n) complexity by using binanry search while comparing a new item to the sorted list. Insertion Sort does 2 things to sort the array: compare an item with the previous items and swap.

This is called **Binary Insertion Sort**. In the end, the complexity for this algo still be O(n^2) but the constant factor for comparation as comparing will cost more than swap. 
- Comparation cost: O(nlogn)
- Swapping cost: O(n^2)
- Worst case: O(n^2)

### 4. Heaps and Heap Sort
Heap: An array visualized as a complete binary tree

Application: priority queue, heap sort,...

- parent(i) = i/2
- left(i) = i*2 + 1
- right(i) = i*2 + 2 (if existed)

Max heap: parent >= its children

Pre-condition of heapify at a node: subtree rooted at its left and right children are max-heaps

=> Heap build routine starts from the end towards the begin -> more exactly it starts from node n/2 - 1 to avoid unnecessary work on leaf nodes. 

### 5. Binary Search Trees
https://youtu.be/9Jry5-82I68
Augumented BST: add extra key in to each node that holds number of nodes below it. Modify when inserting or deleting. Similar approach is used to buid balanced BST AVL by storing height of substree instead of number of nodes

### 6. AVL Trees, AVL Sort
Height of tree = length of the longest path from root down to a leaf \
Height of a node = length of the longest path from it down to a lead

height(node) = max(height(left), height(right)) + 1

AVL invariant: height of left & right children of every node to differ by at most +/- 1
=> maintain the balance.

AVL Sort:
- Insert n elements into tree -> O(nlogn)
- in-order traversal -> O(n)

### 7. Counting Sort, Radix Sort, Lower Bounds for Sorting
To sort big array with small integers in linear time. 

#### Counting sort
Sort a list of integers have value between [0, k]. By using an empty array of k elemnts, and mark found emelements by +1 at a[i].

Final step is to iterate the k array to form the output.

```python
L = [0] * (k + 1) 
for i in range(n):
    L[a[i]].append(a[i])

output = []
for i range(k+1):
    output += L[i] 
```

=> Time complexity = O(n + k)

#### Radix sort
Sort a list of integers

https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec07.pdf

**Step 1. Decompose each integer into set of digits as base b**

For example:
```python
num = 10
digits = []
while num != 0:
    digits.append(num%b)
    num /= b
```
num of digits of an integer k = `d` = log<sub>b</sub>(k) + 1

Time complexity = O(n * d) = O(n * O(log<sub>b</sub>(k))

-> The key of this step is to find `d`. Can interate the array to find max and decompose d from max -> O(n). Then extract digit by mod in O(1)

**Step 2. Sort all integers by least significant digit (most left digit),..., sort all integers by most significant degit (most right digit)**

```python
for i in range(d): 
    for j in range(n):
        # countingSort a[0][i] -> a[n-1][i]

```

Sort using counting sort to sort and list of integers having value between [0, b]. Time complexity to sort ints by each digits = O(n + b)

Total time for this step = O(d * (n + b)) = O(log<sub>b</sub>(k) * (n + b))

Min. when (b == n) then O(nlog<sub>n</sub>(k))

if (k <= n<sup>c</sup>) then O(n.c) (c: constant)

[Implementation](source/algorithms/radix_sort.py)

### 8. Hashing with Chaining
Direct-access table:
- Keys may not be integers
- giantly wastful memory 

Hash table:
- prehash: maps keys to possitive integers
- Reduce universe of all keys down to reasonable size m for table

Resolve collision with chaining: use an assumption called "Simple uniform hashing" -> each key is likely to be hashed correctly. one key map to one slot 

So the expected length of chain for n keys and m slots = n/m = a if m = O(n) => chain length = 1. So the time to search on hash table with chaining is O(a)

### 9. Table Doubling, Karp-Rabin
Grow table: m -> m'
- Build new table of size m'
- Build new hash function h'
- rehash all items in m to m'

Recommended: m' = 2m => table doubling

Search substring "Karp-Rabin": have substring S and message M. Find P in M
- hash(S) = h
- Loop through M from i = 0 -> len(M)-len(S). At each step, calculates hash of substring [i, i+len(S)-1] = h'
- Compare h and h'
- hash function is using 'Rolling Hash'

### 10. Open Addressing, Cryptographic Hashing
Probling until there is an available slot
- hash(key, 1) -> occupied
- hash(key, 2) -> occupied
- ...........
- hash(key, m-1) -> available

**Insert**: probing till there is available slot - "None" slot \
**Search**: probing till found the key or "None" slot \
**Delete**: search for the key and replace the deleted item with "DeleteMe" flag, this flag is treated as "non-None" slot while searching but "None" while inserting

#### Probing strategies
- Linear hashing: if h(key, i) is occupied -> try h(key,i+1) and so on. 
- Double hashing: if h(key, i) is occupied, do the second hashing with different hash func to find new slot

#### Cryptographic Hashing
One-way hashing: having q = hash(x), it's very hard to find x from q

### 13. Breadth-First Search (BFS)
#### Graph
```
G = (V.E)
```
- V: set of vertices
- E: set of edges

Graph Application:
- Web crawling
- Social networking
- Network broadcast
- Garbage collection
- Model checking
- Solve rubic cube 

Presentation of graph: 
- Adjacency list
```
{
    a : {b, c},
    b : {a, c},
    c : {a, b, d}
}
```
{b,c} is the degree of a, sume of all degrees of all nodes we will have 2E (handshaking lemma)
The complexity to travel all nodes and their degrees will be O(V + 2E)
At each node v, we travel v + v's degrees 
So the complexity =  sum(v<sub>i</sub> + degree(v<sub>i</sub>)) => O(V + sum(degree(V))) = O(V + 2E) = O(V + E)

Space Complexity for graph: O(V + E)

- Adjacency matrix
```
  0 1 2 3 
0 0 1 0 0
1 0 1 0 1
2 ...
3 ...
```
We gotta travel from one node to all others even they are not connected :( a[i][j] = 0

Space Complexity for graph: O(V<sup>2</sup>). Can represent in bits as the value is either 0 or 1

Trade off: time vs space

#### BFS
Shortest path in unweighted graph from start node to all the nodes

Time complexity: 
- Adjacency list -> O(V + E). Space O(V + E)
- Adjacency matrix -> O(V<sup>2</sup>). Space: O(V<sup>2</sup>) bits

Undirected graph: each edge contribute 2 degrees to 2 nodes. Total number of degrees of all nodes are 2 times number of edges. 

Directed graph: each node has in degree and out degree. sum of in degrees + sum of out degress = 2.E. `sum of in degrees = sum of out degress` - each edge contribute 1 out degree to one node and 1 in degree to another node. 

-> This is handshaking lemma

**WARNING**: Memorization to avoid duplicates -> avoid running forever. Avoid wrong result on on undirected graph or directed graph with cycle

[Implementation](source/algorithms/bfs.py)
 
Save parent of v to back track for 'a' shortest path (there could be multiple paths) 
```
end > parent[end] > parent[parent[end]] > .... > start
```
The length of the shortest path is levels of the travel. For example:
```
s -> adj[s]...   
      ^ 
reach adjacencies of s in 1 level
```

### 14. Depth-First Search (DFS), Topological Sort
#### DFS
DFS might not find the shortest path due to memorized property of DFS. But it could be possible if we run DFS for the whole graph and update the latest value... -> this seems to be applied only when we tryna find something if it exists :) 

Recursively explore graph, backtracking as necessary OR use stack

Memorize visited nodes to not duplicate. 

[Implementation](source/algorithms/dfs.py)

- forward edge: an edge where a node can access descendant directly. 
- backward edge: an edge where a node can access ancestor directly

{
    a : {b, d},
    b : {c},
    c : {d},
    d : {b}
} \
a -> d: forward edge \
d -> b: backward edge

Child node means a node put later on DFS. For example DFS put `a` to the tree, and later `d`, a->d is a shortcut in the tree

**WARNING** Memorized the visited node to avoid duplication and wrong result...on undirected graph or directed graph with cycle

#### Cycle detection
Graph has a cycle if DSF has a backward edge 

#### Topological Sort
https://youtu.be/AfSk24UTFS8?t=2727

Problem: given directed acyclic (no cycle) graph, order vertices so that all edges point from lower oder to higher order. \
-> Run DFS, output reverse of finishing times of vertices

This is a job scheduling problem. Start visiting by DFS, till we reach the end, so we can back track as the end node (e) is the job needed to be done first and parent[e] > parent[parent[e]]

e has nothing depends on it -> it's safe to start with e first then e's parent

### 15. Single-Source Shortest Paths Problem
- Dijstra: positive weight, no cycles -> O(VlogV + E)
- Bellmen-Ford:  positive/negative weight -> O(V.E). Bellmen-Ford can detect cycle 

Graph with negative weight example: social network, like is positive, dislike is negative


### 16. Dijstra
Implementation is similar to BFS using priority queue(BTS, heap). Have a separete hash to save cost to access with constant time, have another hash to save the parent to back track the path, also we need an hash to check whether a node is processed or not? if one is procecssed already we never do that again. A node while processing update cost for it's neightbors and then mark it as processed :) and jump to process the next lowest cost node (in prority queue)

### 17. Bellmean-Ford


### X. Dynamic Programming
https://www.youtube.com/watch?v=YBSt1jYwVfU 