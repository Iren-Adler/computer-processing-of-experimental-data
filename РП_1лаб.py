"""Пусть дана Z — матрица данных размером N×p.
1) Составить программу для вычисления
а) средних по столбцам, дисперсий по столбцам;
б) стандартизованной матрицы;
в) ковариационной матрицы;
г) корреляционной матрицы.
2) Проверить гипотезу о значимости коэффициентов корреляции
между столбцами матрицы данных."""
from numpy.ma.core import append
from pandas.core.ops.missing import dispatch_fill_zeros

# Чтение данных из файла
f = open("/Users/irayangirova/Downloads/РП_1лаба.txt")
z = []
for line in f:

    numbers = line.strip().split()
    if len(numbers) > 0:
        row = []
        for x in numbers:
            x = x.replace(',', '.')
            row.append(float(x))
        z.append(row)
f.close()


N = len(z)
p = len(z[0])
dig = 4

print('\nСреднее значение j-го признака:')
z_ = []
for j in range(p):
    su = 0
    for i in range(N): su += z[i][j]
    su /= N
    z_.append(su)
    print(f"z_{j} = ", round(su, dig), end = ', ')

print('\n\nОценка дисперсии j-го столбца:')
s2 = []
for j in range(p):
    su = 0
    for i in range(N): su += (z[i][j] - z_[j])*(z[i][j] - z_[j])
    su /= N
    s2.append(su)
    print(f"s2_{j} = ", round(su,dig), end = ', ')

print('\n\nКовариационная матрица (p*p):')
o = [[] for i in range(p)]
for i in range(p):
    for j in range(p):
        su = 0
        for k in range(N):
            su += (z[k][i] - z_[i])*(z[k][j] - z_[j])
        su /= N
        o[i].append(su)
        print(round(su, dig), end=' ')
    print()

print('\n\nОтклонение j-го столбца:')
s = []
for j in range(p):
    su = s2[j]**0.5
    s.append(su)
    print(f"s_{j} = ", round(su,dig), end = ', ')

print('\n\nСтандартизованная матрица Х(N*p):')
x = [[] for i in range(N)]
for i in range(N):
    for j in range(p):
        x[i].append((z[i][j] - z_[j]) / s[j])
        print(round(x[i][j], dig), end=' ')
    print()

print('\n\nКорреляционная матрица R(p*p):')
r = [[] for i in range(p)]
for i in range(p):
    for j in range(p):
        su = 0
        for k in range(N): su += x[k][i]*x[k][j]
        su /= N
        r[i].append(su)
        print(round(r[i][j], dig), end=' ')
    print()

print("\nH0: p(x, y)==0, связи между признаками x и y нет")
print("H1: p(x, y)!=0, то есть связь есть")
print("p(x, y) - коэффициент корреляции между х и у")

print('\n\nt-расчетные:')
t = [[] for i in range(p)]
for i in range(p):
    for j in range(p):
        if i==j:
            t[i].append("N")
            print("N", end=' ')
        else:
            t[i].append(r[i][j] * ((N - 2)**0.5) / ((1 - r[i][j]**2)**0.5))
            print(round(r[i][j], dig), end=' ')
    print()


t_tabl = 1.9960084
print(f'\n\nt_табл = {t_tabl}, a = 0.05, число степ = 69 - 2 = 67')
print('Принятая гипотеза:')
for i in range(p):
    for j in range(p):
        if i == j:
            print("N", end = ' ')
        else:
            if abs(t[i][j]) > t_tabl:
                print("H1", end = ' ')
            else:
                print("H0", end=' ')
    print()
