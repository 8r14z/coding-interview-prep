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
