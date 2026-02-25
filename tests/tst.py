N = int(input())
K = 100
while N > 10:
    P = N % 100
    print(f"P = {N%100}")
    if P < K:
        K = P
    N = N // 10
    print(f"N = {N//10}")
file = open("qantar.txt", 'w')
file.write(str(K))
file.close()