#ТЕСТОВЫЙ ПРИМЕР
"""
x = [[1, 1,  5], [1, 3, 7], [1, 4, 9], [1, 6, 13], [1, 10, 15]]
y = [16, 25, 30, 39, 45]

N = len(x)        # количество наблюдений
m = len(x[0])     # количество признаков


print("Тестовый пример (БЕЗ СВОБОДНОГО ЧЛЕНА)")
print("Исходная матрица X (без свободного члена):")
for i in range(N):
    print(f"   {x[i]}  → y = {y[i]}")
print("=" * 70)


X = x

print(f"\n1. Матрица X (без свободного члена):")
for i in range(N):
    print(f"   {X[i]}")

# 2. Матричные функции
def transpose_matrix(A):
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]

def multiply_matrices(A, B):
    m = len(A)
    n = len(A[0])
    k = len(B[0])
    C = [[0.0 for _ in range(k)] for _ in range(m)]
    for i in range(m):
        for j in range(k):
            s = 0.0
            for t in range(n):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C

def inverse_matrix(A):
    n = len(A)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    augmented = [A[i][:] + I[i] for i in range(n)]

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        pivot = augmented[i][i]
        if abs(pivot) < 1e-10:
            raise ValueError("Матрица вырождена")
        for j in range(2 * n):
            augmented[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    return [row[n:] for row in augmented]

# 3. Вычисление МНК-оценки
print(f"\n2. Вычисление МНК-оценки параметров")

XT = transpose_matrix(X)
print(f"\n   X^T (транспонированная):")
for row in XT:
    print(f"   {row}")

XTX = multiply_matrices(XT, X)
print(f"\n   X^T·X:")
for row in XTX:
    print(f"   {[round(x, 2) for x in row]}")

XTX_inv = inverse_matrix(XTX)
print(f"\n   (X^T·X)⁻¹:")
for row in XTX_inv:
    print(f"   {[round(x, 4) for x in row]}")

XTy = [[0.0] for _ in range(m)]  # размер m, а не m+1
for i in range(m):
    s = 0.0
    for k in range(N):
        s += XT[i][k] * y[k]
    XTy[i][0] = s
print(f"\n   X^T·y:")
print(f"   {[round(x[0], 2) for x in XTy]}")

a = multiply_matrices(XTX_inv, XTy)

print(f"\n3. МНК-оценки параметров регрессии:")
print(f"   y = a1*x1 + a2*x2")
print(f"   a1 (для x1) = {a[0][0]:.6f}")
print(f"   a2 (для x2) = {a[1][0]:.6f}")

# 4. Расчетные значения и проверка качества
y_hat = []
for i in range(N):
    s = 0.0
    for j in range(m):
        s += a[j][0] * X[i][j]
    y_hat.append(s)

e = [y[i] - y_hat[i] for i in range(N)]

# Проверка средних 
y_mean = sum(y) / N
y_hat_mean = sum(y_hat) / N

print(f"\n4. Проверка равенства средних:")
print(f"   Среднее фактическое y = {y_mean:.6f}")
print(f"   Среднее расчетное y_hat = {y_hat_mean:.6f}")
print(f"   Разница = {abs(y_mean - y_hat_mean):.6f}")
print(f"   (для модели без свободного члена средние могут не совпадать)")

# Коэффициент детерминации
TSS = sum((y[i] - y_mean) ** 2 for i in range(N))
RSS = sum(e[i] ** 2 for i in range(N))
R2 = 1 - RSS / TSS

print(f"\n5. Коэффициенты детерминации:")
print(f"   R² = {R2:.6f}")
print(f"   Модель объясняет {R2 * 100:.2f}% дисперсии y")

print(f"\n6. Сравнение фактических и расчетных значений:")
print(f"{'№':>3} {'y_факт':>8} {'y_расч':>8} {'остаток':>10}")
print("-" * 35)
for i in range(N):
    print(f"{i+1:3} {y[i]:8.2f} {y_hat[i]:8.2f} {e[i]:10.6f}")
"""
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
dig = 6



corr_matrix = [
    [1.0, 0.2514, 0.3807, 0.0298, 0.0782, 0.0527, 0.319, -0.0217, 0.2127, 0.2621],
    [0.2514, 1.0, 0.733, 0.3363, -0.2313, -0.1378, -0.0078, -0.2572, 0.1877, 0.4579],
    [0.3807, 0.733, 1.0, 0.3518, -0.4011, -0.3614, -0.1084, -0.3932, 0.0753, 0.3277],
    [0.0298, 0.3363, 0.3518, 1.0, -0.301, -0.187, 0.009, -0.2447, 0.0396, 0.0932],
    [0.0782, -0.2313, -0.4011, -0.301, 1.0, 0.6691, 0.4611, 0.7957, 0.2517, 0.1162],
    [0.0527, -0.1378, -0.3614, -0.187, 0.6691, 1.0, 0.5555, 0.6359, 0.3729, 0.2718],
    [0.319, -0.0078, -0.1084, 0.009, 0.4611, 0.5555, 1.0, 0.4832, 0.3546, 0.4946],
    [-0.0217, -0.2572, -0.3932, -0.2447, 0.7957, 0.6359, 0.4832, 1.0, 0.4371, 0.231],
    [0.2127, 0.1877, 0.0753, 0.0396, 0.2517, 0.3729, 0.3546, 0.4371, 1.0, 0.4975],
    [0.2621, 0.4579, 0.3277, 0.0932, 0.1162, 0.2718, 0.4946, 0.231, 0.4975, 1.0]
]


hypothesis_matrix = [
    ['N', 'H1', 'H1', 'H0', 'H0', 'H0', 'H1', 'H0', 'H0', 'H1'],
    ['H1', 'N', 'H1', 'H1', 'H0', 'H0', 'H0', 'H1', 'H0', 'H1'],
    ['H1', 'H1', 'N', 'H1', 'H1', 'H1', 'H0', 'H1', 'H0', 'H1'],
    ['H0', 'H1', 'H1', 'N', 'H1', 'H0', 'H0', 'H1', 'H0', 'H0'],
    ['H0', 'H0', 'H1', 'H1', 'N', 'H1', 'H1', 'H1', 'H1', 'H0'],
    ['H0', 'H0', 'H1', 'H0', 'H1', 'N', 'H1', 'H1', 'H1', 'H1'],
    ['H1', 'H0', 'H0', 'H0', 'H1', 'H1', 'N', 'H1', 'H1', 'H1'],
    ['H0', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'N', 'H1', 'H0'],
    ['H0', 'H0', 'H0', 'H0', 'H1', 'H1', 'H1', 'H1', 'N', 'H1'],
    ['H1', 'H1', 'H1', 'H0', 'H0', 'H1', 'H1', 'H0', 'H1', 'N']
]


print("\n1. Анализ признаков для выбора зависимой переменной:")

candidates = []
for y_idx in range(p):

    h1_count = sum(1 for j in range(p) if j != y_idx and hypothesis_matrix[y_idx][j] == 'H1')


    corr_count = sum(1 for j in range(p) if j != y_idx and abs(corr_matrix[y_idx][j]) >= 0.3)

    candidates.append((y_idx, h1_count, corr_count))
    print(f"   z_{y_idx}: H1_count={h1_count}, corr>0.3_count={corr_count}")


y_index = max(candidates, key=lambda x: x[1])[0]
print(f"\n   Выбран целевой признак: z_{y_index} (наибольшее количество H1)")

y = [row[y_index] for row in z]


print(f"\n2. Отбор признаков по критерию |r| > 0.3 с z_{y_index}:")
x_indices_corr = []

for j in range(p):
    if j != y_index:
        corr = corr_matrix[y_index][j]
        if abs(corr) >= 0.3:
            x_indices_corr.append(j)
            print(f"   z_{j}: |r| = {abs(corr):.4f} >= 0.3 → ОСТАВЛЯЕМ")
        else:
            print(f"   z_{j}: |r| = {abs(corr):.4f} < 0.3 → УДАЛЯЕМ")

print(f"\n   После фильтрации по |r|>0.3 осталось {len(x_indices_corr)} признаков: z_{x_indices_corr}")


MIN_FEATURES = 2

if len(x_indices_corr) < MIN_FEATURES:
    print(f"\n3. Признаков мало ({len(x_indices_corr)} < {MIN_FEATURES})")

    x_indices = []
    for j in range(p):
        if j != y_index and hypothesis_matrix[y_index][j] == 'H1':
            x_indices.append(j)
            print(f"   z_{j}: H1 → ОСТАВЛЯЕМ")

    print(f"\n   По критерию H1 отобрано {len(x_indices)} признаков: z_{x_indices}")
else:
    print(f"\n3. Признаков достаточно ({len(x_indices_corr)} >= {MIN_FEATURES})")
    print(f"   Используем признаки, отобранные по критерию |r|>0.3")
    x_indices = x_indices_corr

m = len(x_indices)
print(f"\n   Итоговый набор независимых переменных: z_{x_indices}")
print(f"   Количество независимых переменных (m) = {m}")

# 4. Проверка на мультиколлинеарность
print(f"\n4. Проверка мультиколлинеарности:")

high_corr_pairs = []
to_remove = set()

for i in range(len(x_indices)):
    for j in range(i + 1, len(x_indices)):
        idx_i = x_indices[i]
        idx_j = x_indices[j]
        corr = abs(corr_matrix[idx_i][idx_j])
        if corr > 0.8:
            print(f"   Высокая корреляция между z_{idx_i} и z_{idx_j}: {corr:.4f} > 0.8")
            high_corr_pairs.append((idx_i, idx_j, corr))

if high_corr_pairs:
    print(f"\n   Обнаружена мультиколлинеарность. Удаляем признаки:")

    # Для каждой пары удаляем признак с меньшей корреляцией с целевым
    for pair in high_corr_pairs:
        corr_with_y_i = abs(corr_matrix[y_index][pair[0]])
        corr_with_y_j = abs(corr_matrix[y_index][pair[1]])

        if corr_with_y_i < corr_with_y_j:
            if pair[0] not in to_remove:
                to_remove.add(pair[0])
                print(f"   - удаляем z_{pair[0]} (корр. с y={corr_with_y_i:.4f} < {corr_with_y_j:.4f})")
        else:
            if pair[1] not in to_remove:
                to_remove.add(pair[1])
                print(f"   - удаляем z_{pair[1]} (корр. с y={corr_with_y_j:.4f} < {corr_with_y_i:.4f})")

    x_indices = [idx for idx in x_indices if idx not in to_remove]
    m = len(x_indices)
    print(f"\n   Независимые переменные после удаления мультиколлинеарности: z_{x_indices}")
else:
    print(f"   Мультиколлинеарности не обнаружено (все |r| < 0.8)")

# 5. Формирование расширенной матрицы X
X = []
for i in range(N):
    row = [1.0]  # свободный член
    for j in x_indices:
        row.append(z[i][j])
    X.append(row)

print(f"\n5. Сформирована расширенная матрица X размера {N} x {m + 1}")

for i in range(N):
    for j in range(m + 1):
        print(f"{X[i][j]:10.4f}", end=" ")
    print()


# 6. Матричные функции
def transpose_matrix(A):
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def multiply_matrices(A, B):
    m = len(A)
    n = len(A[0])
    k = len(B[0])
    C = [[0.0 for _ in range(k)] for _ in range(m)]
    for i in range(m):
        for j in range(k):
            s = 0.0
            for t in range(n):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def inverse_matrix(A):
    n = len(A)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    augmented = [A[i][:] + I[i] for i in range(n)]

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        pivot = augmented[i][i]
        if abs(pivot) < 1e-10:
            raise ValueError("Матрица вырождена")
        for j in range(2 * n):
            augmented[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    return [row[n:] for row in augmented]


# 7. Вычисление МНК-оценки
print(f"\n6. Вычисление МНК-оценки параметров")

XT = transpose_matrix(X)
XTX = multiply_matrices(XT, X)
XTX_inv = inverse_matrix(XTX)

XTy = [[0.0] for _ in range(m + 1)]
for i in range(m + 1):
    s = 0.0
    for k in range(N):
        s += XT[i][k] * y[k]
    XTy[i][0] = s

a = multiply_matrices(XTX_inv, XTy)

print(f"\n7. МНК-оценки параметров регрессии:")
print(f"   y = a0 + a1*x1 + a2*x2 + ... + a{m}*x{m}")
print(f"   где y - z_{y_index}")
print(f"   a0 (свободный член) = {a[0][0]:.6f}")
for i in range(1, m + 1):
    print(f"   a{i} (для z_{x_indices[i - 1]}) = {a[i][0]:.6f}")

# 8. Расчетные значения и проверка качества
y_hat = []
for i in range(N):
    s = 0.0
    for j in range(m + 1):
        s += a[j][0] * X[i][j]
    y_hat.append(s)

e = [y[i] - y_hat[i] for i in range(N)]

# Проверка средних
y_mean = sum(y) / N
y_hat_mean = sum(y_hat) / N

print(f"\n8. Проверка равенства средних:")
print(f"   Среднее фактическое y = {y_mean:.6f}")
print(f"   Среднее расчетное y_hat = {y_hat_mean:.6f}")
print(f"   Разница = {abs(y_mean - y_hat_mean):.10f}")
print(f"   {'✓ СОВПАДАЮТ' if abs(y_mean - y_hat_mean) < 1e-10 else '✗ НЕ СОВПАДАЮТ'}")

# Коэффициент детерминации
TSS = sum((y[i] - y_mean) ** 2 for i in range(N))
RSS = sum(e[i] ** 2 for i in range(N))
R2 = 1 - RSS / TSS


print(f"\n9. Коэффициенты детерминации:")
print(f"   R² = {R2:.6f}")
print(f"   Модель объясняет {R2 * 100:.2f}% дисперсии y")



