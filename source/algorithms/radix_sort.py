def countingSort(nums: [int], d: int, b: int):
    count = [[]] * b
    for num in nums:
        countIndex = (num // (b ** (d-1))) % b
        if len(count[countIndex]) == 0:
            count[countIndex] = [num]
        else:
            count[countIndex].append(num)

    output = []
    for i in range(b):
        if len(count[i]) > 0:
            output += count[i]

    return output


def radixSort(nums: [int]) -> [int]:
    max = float('-inf')
    base = 10
    for num in nums:
        if num > max:
            max = num
    
    digits = 0
    while max != 0:
        digits += 1
        max //= base

    output = nums
    for d in range(1, digits+1):
        output = countingSort(output, d, base)
        print(nums)

    return output

radixSort([329, 457, 657, 839, 436, 720, 355])