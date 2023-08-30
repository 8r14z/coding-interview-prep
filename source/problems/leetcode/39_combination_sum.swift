// https://leetcode.com/problems/combination-sum/

class Solution {
    func backtrack(_ ans: inout [[Int]], _ combination: inout [Int], _ target: Int, _ idx: Int, _ candidates: [Int]) {
        guard target >= 0 && idx < candidates.count else {
            return
        }

        if target == 0 {
            ans.append(combination)
            return
        }

        for i in idx..<candidates.count {
            guard candidates[i] <= target else {
                continue
            }

            combination.append(candidates[i])
            backtrack(&ans, &combination, target - candidates[i], i, candidates)
            combination.popLast()
        }
    }
    func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
        var ans: [[Int]] = []
        var combi: [Int] = []
        backtrack(&ans, &combi, target, 0, candidates)
        return ans
    }
}