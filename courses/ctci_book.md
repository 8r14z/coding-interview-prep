# Cracking the coding interview

### Big O
Important terms to evaludate efficiency of algorithms
- Time complexity
- Space complexity

**Note**:
- Stack space in recursive calls counts for space complexicity too (how deep the stack call is - height of stack tree at any given time)
- Drop constant factor and keep only dorminant factor. 2N is equivelant to N -> O(N)
- Do this, then do that => O(A+B)
- Do this, and repeatedly do that every interation => O(A*B)
- Amortized time: in worst case happens only once in a while. e.g: Array is doubling size when runs out of capacity. So append will be O(N) in worst case but O(1) in average (amortized time)

### Technique
[Data Structures & Algorithms](../README.md)

Biggest possible number: 2<sup>64</sup> = 18,446,744,073,709,551,616 => 20 digits => radix sort can be pretty much linear :) 

#### Optimization technique
Best quote: "When u get stuck, throw a random hash table" lol
- Precompute and store
- Bottlenecks - Unnecessary work - Duplicated work (BUD)
    - Bottlenecks: the part that slows down the overall runtime. e.g: sort the array before iterating over it. Sorting O(nlogn), Iterating O(n) -> sorting is the bottlenecks
    - Unnecessary work: the part that doesnt really neccessary and makes the overall run time slower.
    - Duplicated work: usually optimize this part will makes the whole algo optimized. e.g instead of iterating the array everytime to find sum of 2 number we can have a hash table to keep track of the num that has been seen
- Transform the problem into a simpler version and solve it, then work up on more complex version
- Solve the base case first :) <- this is a good one. solve more complex problems by reusing the simple problem
- Emergency List lol basically back up ur mind with bunch of data structures and algos, then try one by one

### Array and  Strings
Amortized time complexity of appending into array is O(1) and O(N) for the worst case when array is out of capacity and need to reallocate new block of memory and copy all existing elements

-> We can say that array.append is O(1) in average / amortized complexity 

### Linked list
The runner technique == fast/slow pointers

Simple example: to find mid pointer, have slow point interates through linked list and fast pointer interates double speed, stop when fast pointer hits null => slow is mid

### Trees and Graphs
Definition of trees can be vary slightly with respect to equality.
- Do not have duplicates
- Duplicates are in left
- Duplicates are in right

Kinds of trees:
- Complete binary trees: Every level except the last level is completely filled and all the nodes are left justified.
- Full binary trees: Every node has 0 or two children
- Perfect binary trees: Every node except the leaf nodes have two children and every level (last level too) is completely filled.

Binary tree traversal:
- In-order: L -> N -> R
- Pre-order: N -> L -> R
- Post-order: L -> R -> N

Binary heaps: min/max
- Complete binary tree, each node > children (max-heaps) or < children (min-heaps). The root is min or max value
- Insert to the end and swap it up 
- Extract min: replace root with last node, heapify new root. 

Tries (prefix trees): [implementation](../source/data_structure/trie.py)
- O(k) lookup time with k is input's length 

Graphs: trees are connected, acyclic graphs
