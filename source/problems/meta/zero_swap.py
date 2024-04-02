def zeroSwap(input):
    pIndex = 0 # hold last num that is different from zero
    for i, num in enumerate(input):
        if num == 0:
            input[pIndex], input[i] = input[i], input[pIndex]
            pIndex += 1
    n = len(input)
    input[pIndex], input[n-1] = input[n-1], input[pIndex]
    print(input)


def zeroSwapReverse(input):
    pIndex = 0 # last zero index
    for i, num in enumerate(input):
        if num != 0:
            input[i], input[pIndex] = input[pIndex], input[i]
            pIndex += 1
    n = len(input)
    input[pIndex], input[n-1] = input[n-1], input[pIndex]
    print(input)

zeroSwap([0,1,2,0,3,4])
zeroSwapReverse([0,1,2,0,3,4])



