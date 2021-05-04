def selection_sort(array):
    for i in range(len(array)):
        minIndex = -1
        for j in range(i, len(array)):
            if minIndex == -1 or array[j] < array[minIndex]:
                minIndex = j
        array[i], array[minIndex] = array[minIndex], array[i]
    return array