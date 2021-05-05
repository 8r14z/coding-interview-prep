def find_max_price(price_arr, weight_arr, max_weight):
    max_price = float('-inf')
    start = 0
    cur_weight = 0
    cur_price = 0

    for end in range(len(price_arr)):
        cur_weight += weight_arr[end]
        cur_price += price_arr[end]
        while cur_weight > max_weight:
            cur_weight -= weight_arr[start]
            cur_price -= price_arr[start]
            start += 1

        max_price = max(cur_price, max_price)

    return max_price

print(find_max_price([3000, 3000, 2000, 1500], [36, 30, 20, 15], 35))