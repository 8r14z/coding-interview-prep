# calculate number of ways
# min/max numbe of coins will similar to unbouned knapsack [implement](./knapsack.py)

def num_of_ways(coins, n):
    m = len(coins)
    dp = [[0]*(n+1) for _ in range(m)]

    for i in range(m):
        for j in range(n+1):
            if j == 0: 
                dp[i][j] = 1
                continue
            
            # to avoid duplicated set like {1,2} and {2,1} for 3
            # at each coin we ask a question whether we should include the coin or not
            # similar to knapsack 0/1
            num_including_coin_set = (dp[i][j-coins[i]]) if (j >= coins[i]) else 0
            num_excluding_coin_set = dp[i-1][j] if i >= 1 else 0

            dp[i][j] = num_including_coin_set + num_excluding_coin_set
    
    return dp[m-1][n]
 
# this particular problem need a coin value == 1 to be solvable
def minimum_num_of_coins(coins, n):
    m = len(coins)

    dp = [0] * (n+1)
    
    for i in range(1, n+1):
        min_way = float('inf')
        for j in range(m):
            if coins[j] <= i:
                min_way = min(min_way, dp[i-coins[j]] + 1)
        dp[i] = min_way
    
    return dp[n]


print(num_of_ways([1,2,3], 4)) # 4
print(num_of_ways([1, 5, 10], 12)) # 4
print(num_of_ways([2, 5, 3, 6], 10)) # 5

print(minimum_num_of_coins([1,2,5], 11))