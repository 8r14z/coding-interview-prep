# merge routine
def merge(a1, a2):
    newArray = []
    i = 0
    j = 0
    
    while i < len(a1) or j < len(a2):
        if i >= len(a1):
            newArray.append(a2[j])
            j += 1
        elif j >= len(a2):
            newArray.append(a1[i])
            i += 1
        else:
            if a1[i] < a2[j]:
                newArray.append(a1[i])
                i += 1
            else:
                newArray.append(a2[j])
                j += 1

    return newArray

def merge_sort(a):
    if a == None: return []

    n = len(a)
    if n < 2: return a

    mid = (n-1)//2
    a1 = merge_sort(a[0 : mid+1])
    a2 = merge_sort(a[mid+1 : len(a)])
    return merge(a1, a2)

print(merge_sort([1,25,21,1,9,10,17]))