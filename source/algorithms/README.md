Yes, radix sort is always linear for constant length keys (constant number of "digits"). You need only 8 passes over the data to sort any number of 32 bit keys with radix-256 sort for instance. And only two passes if you use radix-232 (but this is impractical, see below).

Radix sort can only work if you can extract individual digits from the keys, and you can order them based on their value alone (without comparisons). Sorts that work like this are generally O(m * n), where m is the maximum length of the keys.

Mergesort and other well known sorting algorithms don't have access to that information, and so these algorithms are "comparison sorts", and cannot be any faster then O(m * n log n) in the worst case (but can be significantly faster in the best case).

But these are theoretical results.

In practice there are several considerations:

Quicksort is an in-place sort and so does not require any extra memory.

Straight quicksort is unreliable - due it its worst case performance of O( n2 ). So most programming libraries expose a hybrid variant. This can be as simple as quicksort and heapsort stapled together to form a sort that's guaranteed to be O(n log n).

Quicksort is adaptive - a good implementation can use preexisting order in the data to run faster (note - a naive implementation goes quadratic with preexisting order). But this can also be a curse - implementing a quicksort that is fast under various degenerate input files is actually quite difficult.

Quicksort is extremely cache efficient. Every memory access is sequential.

Key indexed counting is highly cache inefficient due to the random nature of the memory accesses.

Radix sort requires access to individual digits - which makes it harder to use.

Top down radix-2 sort is a special case. It can use the regular partition function from quicksort instead of key indexed counting, which tends to be much more efficient because it only needs to access the data once (and in order).

Radix-k sort where k > 2 requires extra memory in the order O(k) or O(n+k) depending on the implementation of the key indexed counting sort (in place or not).