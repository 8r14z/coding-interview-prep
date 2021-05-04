def binary_search(list, key):
    low = 0
    high = len(list) - 1

    while low <= high:
        mid = (low + high) // 2
        cur_value = list[mid]

        if cur_value == key:
            return mid
        if cur_value > key:
            high = mid - 1
        else:
            low = mid + 1

    return -1
    
print(binary_search([1,2,3,4,5], 6))