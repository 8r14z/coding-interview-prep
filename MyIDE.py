class FindSumPairs:

    def makeHashCount(self, ints: [int]) -> dict:
        tmp = {}
        n = len(ints)
        for i in range(n):
            if ints[i] not in tmp:
                tmp[ints[i]] = 1
            else:
                tmp[ints[i]] += 1
        return tmp
        
    def __init__(self, nums1: [int], nums2: [int]):
        self.nums2 = nums2
        self.hashNums1 = self.makeHashCount(nums1)
        self.hashNums2 = self.makeHashCount(nums2)

    def add(self, index: int, val: int) -> None:
        preValue = self.nums2[index]
        postValue = preValue + val
        self.nums2[index] = postValue
        
        self.hashNums2[preValue] -= 1
        if postValue not in self.hashNums2:
            self.hashNums2[postValue] = 0
        self.hashNums2[postValue] += 1
        
    def count(self, tot: int) -> int:
        count = 0
        
        for num1 in self.hashNums1:
            if tot-num1 in self.hashNums2:
                count += (self.hashNums1[num1] * self.hashNums2[tot-num1])
    
        return 0

findSumPairs = FindSumPairs([1, 1, 2, 2, 2, 3], [1, 4, 5, 2, 5, 4]);
findSumPairs.count(7);