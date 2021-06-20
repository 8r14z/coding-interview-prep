def partition(array, left, right):
    pivot = array[right]
    p_index = left  # p_index is the last position where array[i] > pivot
    for j in range(left, right): # left -> right - 1
        if array[j] <= pivot:
            array[p_index], array[j] = array[j], array[p_index]  # swap arr[i] arr[j]
            p_index += 1

    array[p_index], array[right] = array[right], array[p_index]
    return p_index
    
def findKthLargest(arr, k):
    left = 0
    right = len(arr) - 1

    while left <= right:
        pivotIndex = partition(arr, left, right)

        if pivotIndex == len(arr) - k:
            return pivotIndex
        elif pivotIndex > len(arr) - k:
            right = pivotIndex - 1
        else:
            left = pivotIndex + 1
    return -1

print(findKthLargest([5,7,2,3,4, 1,6], 3))
print(partition([1,2,3,4],0,3))