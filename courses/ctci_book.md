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

All algo needed for trees and graphs: DFS, BFS, Bidirectional BFS (more optimal BFS in searching in undirected graph), inorer, preorder, postorder traversals, Dijkstra, Bellman-Ford, Floyd-Warshall

### Bit manipulation
- Two's complement presentation. N bits, the most significant bit is used to maintain sign. 0 for positive and 1 for negative. N bits, 1 bit of sign, N-1 bits are for value. Thus 32 bits signed numbers will be in [-2^31, 2^31-1] (-1 is for zero)
- Logical right shift will fill with 0 (>>>)
- Arithmetic righ shift will fill with sign bit (>>)
- Unsigned right shift will fill with 0 
- Signed right shift will fill with sign bit
- All left shift fill 0 from the left

### Object-Oriented Design
Approach:
1. Clarify requirements - 6Ws technique What/Why/Who/When/Where/How
2. Define core objects that's critical to the problem statement. For example, to design a restaurant, the core objects will be Guest, Employee, Table, Order, Menu, ...
3. Define relationships between objects. E.g N-to-N, 1-to-N, 
4. Define methods/actions each objects
 
### Recursion and DP


### System Design
1. Clarify/scope requirements
2. Make reasonable assumptions. Assumptions are what define the foundation of a problem. U working on a system and assume that its dependencies are well done. Assume on volume of traffic, assume data consumption, etc
3. Define core components: Client, Server, DB, etc.
4. Sketch out user flows
5. Identify the component to focus on. Bottlenecks, major challenges, performance issues, what component is heavily used, etc 
6.  State trade-offs and limitation in current design

#### Vertical vs. Horizontal Scaling
Vertical: increase the computing power of one node
Horizontal: increase number of nodes

#### Load balancer
Allows to distribute loads evenly to multiple machines (servers). To make sure the workload on one machine reasonable and maintain the heath of server. 

#### SQL vs NoSQL
Join operations in SQL will get slow as the system grows bigger
- Normalization is to avoid redundancy, duplicated data between tables. Tables establish relationship with each other, to get combined data, use joins -> bottlenecks in the long run
- Denormalization, meaing adding redundant tables to avoid join operations. For example, tables Orders and Users, to avoid joining tables once querying orders of a users, we can have a table for such mapping called UserOrder -> Costly for write operations, risk of inconsistent data, more storage

#### DB partitioning (Sharding)
Slit data into different databases/machines
- Vertical sharding: partitioning by feature/projects. One for User Profiles, one for Messages, etc
- Key-based (Hash-based) sharding: paritioning by some sort of data. For examplem, partition Users by Users.location. Partitioned by hashed of a field. 
- Directory-based sharding: maintain lookup table for where data can be found -> Single point of failure (SPOF), performance issues with single table accesses

#### Caching
#### Asynchronous Processing and Queues
#### Network metrics
- Bandwidth: max amount of data can be transferred in a unit of time
- Throughput: the actual amount of data that is transferred
- Latency: how long it takes to go from one to the other. The delay between sender and receiver
#### MapReduce system
This allows us to process a lot of processing in parallel, which makes processing huge amount of data more efficient 

#### System considerations
- Failures. Single point of failures
- Availability (Uptime, Mornitoring & Alerting, Logging) and Reliability
- Scalability
- Read-heavy (caching) and Write-heavy (queue write, data consistency, concurrency)
- Security (Fraud and Abuse, Data Privacy, Authentication)
- Pre-computation







