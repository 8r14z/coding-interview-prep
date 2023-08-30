// https://leetcode.com/problems/rotting-oranges/

class Solution {
    let empty = 0
    let fresh = 1
    let rotten = 2
    let visited = 3

    func orangesRotting(_ grid: [[Int]]) -> Int {
        let m = grid.count
        assert(m > 0)
        let n = grid[0].count
        assert(n > 0)

        var visitedCells: Set<[Int]> = []
        var queue: [[Int]] = []

        var freshCount = 0
        var minutes = 0
        for i in 0..<m {
            for j in 0..<n {
                if grid[i][j] == fresh {
                    freshCount += 1
                } else if grid[i][j] == rotten {
                    visitedCells.insert([i,j])
                    queue.append([i,j])
                }
            }
        }

        while queue.count > 0 {
            var newQueue: [[Int]] = []

            for coordinator in queue {
                let x = coordinator[0]
                let y = coordinator[1]
                let directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
                for direction in directions {
                    let nextX = x + direction[0]
                    let nextY = y + direction[1]
                    guard nextX >= 0 && nextX < m && nextY >= 0 && nextY < n else {
                        continue
                    }

                    guard grid[nextX][nextY] == fresh else {
                        continue
                    }
                    
                    let newCoordinator = [nextX, nextY]
                    guard !visitedCells.contains(newCoordinator) else {
                        continue
                    }

                    freshCount -= 1
                    newQueue.append(newCoordinator)
                    visitedCells.insert(newCoordinator)
                }
            }

            queue = newQueue
            if queue.count > 0 {
                minutes += 1
            }
        }

        return freshCount == 0 ? minutes : -1
    }
}