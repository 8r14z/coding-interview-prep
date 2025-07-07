# Algorithms: Practical Considerations

This document summarizes practical considerations and insights for common algorithms, especially sorting and related concepts.

---

## Quicksort

- **In-place:** Quicksort sorts the array in place and does not require extra memory.
- **Worst-case performance:** Straightforward quicksort can degrade to O(n²) in the worst case. Most libraries use a hybrid approach (e.g., combining quicksort and heapsort) to guarantee O(n log n) performance.
- **Adaptivity:** A well-implemented quicksort can take advantage of preexisting order in the data to run faster. However, naive implementations may perform poorly on already sorted data.
- **Cache efficiency:** Quicksort is extremely cache-efficient because it accesses memory sequentially.

## Key-Indexed Counting & Radix Sort

- **Key-indexed counting:** This method is generally cache-inefficient due to random memory access patterns.
- **Radix sort:** Requires access to individual digits, which can make it less flexible. Top-down radix-2 sort can use the partition function from quicksort, improving efficiency by accessing data sequentially.
- **Radix-k sort (k > 2):** Needs extra memory, either O(k) or O(n + k), depending on whether the key-indexed counting sort is in-place.

## Heapsort

- **Stability:** Heapsort is a stable sort with O(n log n) time complexity.
- **Cache behavior:** It has a higher constant factor due to the randomness of swaps, making memory access patterns less predictable. However, this is usually a minor concern in practice.

---

## Locality of Reference

- **Temporal locality:** If a memory location is accessed, it is likely to be accessed again soon.
- **Spatial locality:** If a memory location is accessed, nearby locations are likely to be accessed soon as well.

---

## Recursion: Tail vs. Non-tail

- **Non-tail recursion:** The recursive call is not the last operation; results are computed after the call returns. This can lead to deep call stacks.
- **Tail recursion:** The recursive call is the last operation. Calculations are done before the call, and the result is passed along. Compilers can optimize tail recursion to reuse stack frames, preventing stack overflow.

---

## Other Concepts

### Rolling Hash Function

- Computes a hash value based on the previous hash, enabling efficient substring search (e.g., Rabin-Karp algorithm).

### MapReduce

- Involves a Map step and a Reduce step. Input data is split across multiple machines to enable parallel processing.
