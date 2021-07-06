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
- 

