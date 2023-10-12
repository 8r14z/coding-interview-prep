// https://leetcode.com/problems/unique-paths/description/
// time: O(m*n)
// space: O(n)
class Solution {
    func uniquePaths(_ m: Int, _ n: Int) -> Int {
        var dp = Array(repeating: 1, count: n)

        for i in 1..<m {
            for j in 0..<n {
                if j == 0 {
                    dp[j] = 1
                } else {
                    dp[j] = dp[j-1] + dp[j]
                }    
            }
        }

        return dp[n-1]
    }
}