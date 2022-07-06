# Everything You Need For Coding Interviews

### Data Structure and Algorithms Courses
- [MIT 6.006 Introduction to Algorithms](courses/mit_6006.md)
- [MIT 6.046J Design and Analysis of Algorithms](courses/mit_6046.md)
- [Cracking the coding interview](courses/ctci_book.md)
- [geeksforgeeks - Data Structures](https://www.geeksforgeeks.org/data-structures/)
- [geeksforgeeks - Algorithms](https://www.geeksforgeeks.org/fundamentals-of-algorithms/)
- [techinterviewhandbook.com](https://www.techinterviewhandbook.org/)

```
2 geeksforgeeks links have most important things
```

### Youtube Resources
- [Graph Theory Playlist](https://www.youtube.com/playlist?list=PLDV1Zeh2NRsDGO4--qE8yH72HFL1Km93P)
- [Hackerrank - Data Structures](https://www.youtube.com/playlist?list=PLI1t_8YX-Apv-UiRlnZwqqrRT8D1RhriX)
- [Hackerrank - Algorithms](https://www.youtube.com/playlist?list=PLI1t_8YX-ApvMthLj56t1Rf-Buio5Y8KL) 
- [A playlist combining videos from 2 Hackerrank playlists](https://www.youtube.com/playlist?list=PLX6IKgS15Ue02WDPRCmYKuZicQHit9kFt)
- [Software Engineering Interviews](https://www.youtube.com/playlist?list=PLiQ766zSC5jPIKibTa5qtXpwgwEBalDV4)
- [Harvard COMPSCI 224 Advanced Algorithms](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uP4rJgf5ayhHWgw7akUWSf)

### Cheatsheet
To quickly revisit algorithms, data structures and their time/space complexity
- [Big-0 Cheatsheet](https://www.bigocheatsheet.com/)
- [Algorithms & Data strucuture Cheatsheet](https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/)

### Pick A Language and Stick With It
- I'd strongly suggest [Python](https://www.geeksforgeeks.org/python-programming-language/?ref=ghm) due to its simplicity and syntax sugar. Also, it supports a wide range of built-in data structures which are super convenient for coding interviews

#### Data Structures
- Array
- Stack & Queue 
- LinkedList 
- Heaps
- Trees, BST/Balanced BST, Tries
- Hash Table
- Graphs

#### Algorithms
- Sorting/Searching/DFS/BFS/Tree traversal

#### Technique 
- Brute-force
- Divide and Conquer 
- Dynamic Programming (DP)
- Recursion / Backtracking
- Bit manipulation
- Tree traversal

#### [Patterns](https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed)
1. Sliding Window
To perform an operation on a specific/dynamic window size. Start from 1st on the left and keep shifting to right

2. Two pointers
Keep two pointers in an array or linked list. Usually it's head and tail. Keep moving/modifying 2 pointers to achieve more efficient solution

3. Fast & slow pointers
A special kind of 2 pointers. Fast pointer's speed is 2x of slow pointer. Typically to detect cyclic sequence

4. Merge intervals
A popular problem of this kind of problenm is time scheduling. to merge time that overlap with each others. Usually can be solved by greedy alogrithm

5. Linked list manipulation
There are a wide range of problems that involves linked list and manipulate nodes on linked list. A popular example could be reversing linked list (this is almost a meme lol)

6. DFS and BFS on trees/graphes
They are 2 most important algorithms in set of trees and graphes, appeared in almost every tree/graph problems

7. Using heap to find min/max
Heap is an important data structure, it's key for priority queue and most problems related to scheduling or optimize timeline will involve heap.

8. Prefix sum: where we can optimize an result by previously calculated results in a sequence 

### Practical Tips To Solve Any Problems
1. Practice
2. Practice more 
3. Repeat 

```
On a side note: Try applying Hash Table / Graphs / Stack & Queue / Heap / Binary Tree when getting stuck

Most important algirthms & data structures are DFS/BFS, Heap (heapify), Hash Table
```

Try to find a way to improve time complexity. Common time complexity (sorted fast -> slow)
- O(1)
- O(log(n))
- O(n)
- O(n*log(n))
- O(n^2)
- O(n^3) <-- this is already the worst you are able to find except some NP-hard problems which are n^n
