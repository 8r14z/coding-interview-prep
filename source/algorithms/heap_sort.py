def heapify(array, n, i):
    while i < n//2:
        child = i*2 + 1
        if child < n-1 and array[child] < array[child+1]:
            child += 1
        
        if array[i] > array[child]:
            break

        array[i], array[child] = array[child], array[i]
        i = child

def buildHeap(array):
    n = len(array)
    for i in range(n//2-1, -1, -1):
        heapify(array, n, i)

def heapSort(array):
    buildHeap(array)
    n = len(array)

    for i in range(n-1, 0, -1):
        array[0], array[i] = array[i], array[0]
        heapify(array, i, 0)

    return array

print(heapSort([10,6,7,5,15,17,12])) # [5, 6, 7, 10, 12, 15, 17] 

# Build heap is O(n)
# https://www.geeksforgeeks.org/time-complexity-of-building-a-heap/

# https://stackoverflow.com/questions/9755721/how-can-building-a-heap-be-on-time-complexity
# Intuitively, we can see infer the buildHeap algorithm as nlog(n), it's a upper bound but not tight bound
# At cost of heapify at each step is different, depending on the high of subtree which is different at different levels
# At leaf-node level, there are n/2 nodes and the high = 0 => cost = n/2 * 0
# At upper level, there are n/2/2 = n/4 nodes and high = 1 => cost = n/4 * 1
# ...
# At the parent node, there is 1 node and high = log(n) = heigh of tree (h) => cost = 1 * h
# If we keep moving upward the tree and do the heapify, we can build the series as
# n/2 * 0 + n/4 * 1 + n/8 * 2 + ... + 1 * h
# = h * n / 2^(h+1) with h range from (0, logn)