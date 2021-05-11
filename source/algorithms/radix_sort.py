import math
def countingSort(nums: [int], d: int, base: int):
    
    # count = [[]] * base

    # for num in nums:
    #     countIndex = (num // (base ** (d-1))) % base
    #     if len(count[countIndex]) == 0:
    #         count[countIndex] = [num]
    #     else:
    #         count[countIndex].append(num)

    # output = []
    # for i in range(base):
    #     if len(count[i]) > 0:
    #         output += count[i]

    # return output

    # improvement - in-place position array
    # [3,2,5,1,5,3,4] -> sorted [1,2,3,3,4,5,5]
    # 1 - build pos array with num of occurrence of on each index
    # [0,1,1,2,1,2]
    # 2 - inplace transform to array of num of items that less than pos[i]
    # [0,0,1,2,4,5] -> this is also the start index in the sorted array
    # for example 5 in sorted array is started at index 5 
    #  0 1 2 3 4 5 6 <- index
    # [1,2,3,3,4,5,5]
    # in the output array, get output[pos[num]] = num and increment post[num] by 1, 
    # so the next time the same num is selected it will be slotted in the next index
    # for example: output[pos[5]] = 5, pos[5] += 1
    #                      ^            ^
    #                      5            6
    # https://youtu.be/9bkvws_vqLU?t=1537
    
    n = base
    pos = [0] * n
    
    for num in nums: 
        countIndex = (num // (base ** (d-1))) % base
        pos[countIndex] += 1
    
    numOfNums = len(nums)
    for i in range(n-1, -1, -1):
        numAtPos = numOfNums - pos[i]
        pos[i] = numAtPos
        numOfNums = numAtPos

    output = [0] * len(nums)
    for num in nums:
        countIndex = (num // (base ** (d-1))) % base
        output[pos[countIndex]] = num
        pos[countIndex] += 1
    
    return output


def radixSort(nums: [int]) -> [int]:
    max = float('-inf')
    for num in nums:
        if num > max:
            max = num
    
    digits = int(math.log10(max)) + 1

    for d in range(1, digits+1):
        nums = countingSort(nums, d, 10)
        print(nums)

    return nums


radixSort([329, 457, 657, 839, 436, 720, 355])