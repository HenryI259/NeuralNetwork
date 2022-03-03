def vectorResult(i):
    zeros = [0 for x in range(10)]
    zeros[i-1] = 1.0
    return zeros

def shift(l, shiftBy=1):
    newList = [i for i in range(len(l))]
    for item in newList:
        index = item - shiftBy
        while index < 0:
            index += len(l)
        while index >= len(l):
            index -= len(l)
        newList[item] = l[index]
    return newList