# [https://www.khanacademy.org/computing/computer-science/cryptography/modarithmetic/a/the-euclidean-algorithm]

def greatestCommonDivisor(a, b):
    if a == 0:
        return b
    if b == 0:
        return a

    remainder = a % b
    return greatestCommonDivisor(b, remainder)

print(greatestCommonDivisor(160, 70))