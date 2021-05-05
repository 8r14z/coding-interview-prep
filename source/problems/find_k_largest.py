def partition(array, left, right):
    pivot = array[right]
    i = left  # i is the last position where arra[i] > pivot
    for j in range(left, right):
        if array[j] <= pivot:
            array[i], array[j] = array[j], array[i]  # swap arr[i] arr[j]
            i += 1

    array[i], array[right] = array[right], array[i]
    return i
    
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