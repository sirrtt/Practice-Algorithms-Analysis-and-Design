def nhap():
    p=[1]
    n=int(input())
    for i in range(n): p.append(int(input()))
    p.append(1)
    ketqua=[[]for _ in range(n+2)]
    for i in range(n+2):
        for j in range(n+2): ketqua[i].append(-1)
    return (n, ketqua, p)

def tim(ketqua, p, phai, trai):
    if(ketqua[trai][phai]!=-1): return ketqua[trai][phai]
    tien=0
    for i in range(trai+1, phai): tien=max(tien, p[trai]*p[i]*p[phai]+tim(ketqua, p, i, trai)+tim(ketqua, p, phai, i))
    ketqua[trai][phai]=tien
    return ketqua[trai][phai]

n, ketqua, p = nhap()
print(tim(ketqua, p, n+1, 0))
