In practice there are several considerations:
- Quicksort is an in-place sort and so does not require any extra memory.
- Straight quicksort is unreliable - due it its worst case performance of O( n2 ). So most programming libraries expose a hybrid variant. This can be as simple as quicksort and heapsort stapled together to form a sort that's guaranteed to be O(n log n).
- Quicksort is adaptive - a good implementation can use preexisting order in the data to run faster (note - a naive implementation goes quadratic with preexisting order). But this can also be a curse - implementing a quicksort that is fast under various degenerate input files is actually quite difficult.
- Quicksort is extremely cache efficient. Every memory access is sequential.
- Key indexed counting is highly cache inefficient due to the random nature of the memory accesses.
- Radix sort requires access to individual digits - which makes it harder to use.
- Top down radix-2 sort is a special case. It can use the regular partition function from quicksort instead of key indexed counting, which tends to be much more efficient because it only needs to access the data once (and in order).
- Radix-k sort where k > 2 requires extra memory in the order O(k) or O(n+k) depending on the implementation of the key indexed counting sort (in place or not).


#### Locality of reference
- Temporal locality: If at one point a particular memory location is referenced, then it is likely that the same location will be referenced again in the near future.
- Spatial locality: If a particular storage location is referenced at a particular time, then it is likely that nearby memory locations will be referenced in the near future. 

Heapsort is a stable sort with O(nlogn) but it has big constant factor based on the randomness of picking items to swap. It makes the operating hard to predict what memory location that u should jump to next. But anyway this is minor 

### Tail/Non-tail Recursion
In traditional recursion, the typical model is that you perform your recursive calls first, and then you take the return value of the recursive call and calculate the result. In this manner, you don't get the result of your calculation until you have returned from every recursive call.

In tail recursion, you perform your calculations first, and then you execute the recursive call, passing the results of your current step to the next recursive step. This results in the last statement being in the form of (return (recursive-function params)). Basically, the return value of any given recursive step is the same as the return value of the next recursive call.

The consequence of this is that once you are ready to perform your next recursive step, you don't need the current stack frame any more. This allows for some optimization. In fact, with an appropriately written compiler, you should never have a stack overflow snicker with a tail recursive call. Simply reuse the current stack frame for the next recursive step.

#### Rolling hash function
To compute hash of based on existing hash value. Used for Rabin-Karp searching :) 

#### MapReduce
Includes Map step and Reduce step and input is splitted into multiple machines to enable parallel execution
