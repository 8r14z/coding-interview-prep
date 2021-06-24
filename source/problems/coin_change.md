# calculate number of ways
# otherwise it will similar to unbouned knapsack [implement](./knapsack.py)

def coin_change(coins, amount):
    n = len(coins)
    dp = [0] * (amount+1)

    for i in range(1, amount+1):
        count = 0
        for j in range(n):
            change = i - coins[j]
            if change > 0:
                if change == 1:
                    count += 1
                else:
                    count += dp[change]
        dp[i] = count
        
    return dp[amount]

from functools import cache
def coin_change_topdown(coins, amount):
    @cache
    def dp(amount):
        if amount == 1: return 1

        sum = 0
        for i in range(len(coins)):
            if coins[i] <= amount:
                sum += dp(amount-coins[i])
        return sum
    return dp(amount)

print(coin_change([1,2,3], 4))
print(coin_change_topdown([1,2,3], 4))