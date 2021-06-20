def partition(array, left, right):
    pivot = array[right]
    i = left  # i is the last position where arra[i] > pivot
    for j in range(left, right):
        if array[j] <= pivot:
            array[i], array[j] = array[j], array[i]  # swap arr[i] arr[j]
            i += 1

    array[i], array[right] = array[right], array[i]
    return i


def quicksort_helper(array, left, right):
    if left < right:
        pivotIndex = partition(array, left, right)
        quicksort_helper(array, left, pivotIndex - 1)
        quicksort_helper(array, pivotIndex + 1, right)

def quicksort(array):
    quicksort_helper(array, 0, len(array) - 1)

