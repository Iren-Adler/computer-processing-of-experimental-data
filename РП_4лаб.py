"""
Лабораторная работа: прогнозирование временного ряда
ВАРИАНТ 21
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

X = np.array(
    [43, 44, 45, 42, 47, 48, 49, 53, 56, 60, 66, 72, 73, 77, 81, 78, 79, 87, 94, 93, 84, 92, 100, 106, 110, 108, 111,
     103, 109, 121], dtype=float)
N = len(X)  # N = 30
weeks = np.arange(1, N + 1)  # t = 1, 2, ..., 30


out_dir = Path('../lab_results')
out_dir.mkdir(exist_ok=True)


print("Прогнозирование временных рядов")
print("Вариант 21 - данные продаж за 30 недель")
print("Оценки математического ожидания и дисперсии")

# Оценка математического ожидания (среднее арифметическое)
mean_X = np.sum(X) / N
print(f"M[x] = {mean_X:.4f}")

# Оценка дисперсии (несмещенная) - формула: s^2 = 1/(n-1) * Σ(x_i - x̄)^2
var_X = np.sum((X - mean_X) ** 2) / (N - 1)
print(f"  Оценка дисперсии: D[x] = {var_X:.4f}")
print(f"  Стандартное отклонение: σ = {np.sqrt(var_X):.4f}")

# Построение графика исходного временного ряда
plt.figure(figsize=(12, 6))
plt.plot(weeks, X, 'b-o', linewidth=1.5, markersize=4)
plt.title('Рис.1. Исходный временной ряд x(t), Вариант 21', fontsize=14)
plt.xlabel('Неделя (t)', fontsize=12)
plt.ylabel('x(t) - объем продаж', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / '01_original_series.png', dpi=150)
plt.show()


# Постоянная модель (n=0) для прогнозирования

print("\n" + "=" * 60)
print("ПУНКТ 2. Постоянная модель (n=0)")


# Параметры сглаживания
alpha_values = [0.1, 0.3]
beta = lambda a: 1 - a  # β = 1 - α

# Начальное значение S0[1] как среднее арифметическое пяти первых значений
S0_const = np.mean(X[:5])
print(f"S0[1] = {S0_const:.4f}")

const_results = {}

for alpha in alpha_values:
    S = np.zeros(N + 1)  # S[1]_t, t = 0..N
    S[0] = S0_const

    pred = np.zeros(N)  # прогноз x_hat_t
    error = np.zeros(N)  # ошибка Δx_t = x_t - x_hat_t

    # Рекуррентное сглаживание: S_t[1] = α*x_t + (1-α)*S_{t-1}[1]
    for i in range(1, N + 1):
        pred[i - 1] = S[i - 1]  # прогноз на момент t (x_hat_t = S_{t-1}[1])
        error[i - 1] = X[i - 1] - pred[i - 1]  # Δx_t = x_t - x_hat_t
        S[i] = alpha * X[i - 1] + (1 - alpha) * S[i - 1]

    # Дисперсия ошибки по формуле (10): s^2[x_hat_t] = Σ(x_t - x_hat_t)^2 / (N - n - 1)
    # n = 0, поэтому знаменатель = N - 1
    var_error_const = np.sum(error ** 2) / (N - 1)

    # Прогноз на следующую неделю (t = N+1)
    future_pred_const = S[N]

    const_results[alpha] = {
        'S': S, 'pred': pred, 'error': error,
        'var_error': var_error_const, 'future_pred': future_pred_const
    }

    print(f"Дисперсия ошибки: s² = {var_error_const:.4f}")
    print(f"Прогноз на 31-ю неделю: x_hat_31 = {future_pred_const:.2f}")

    # Таблица результатов
    print("\nТаблица результатов (первые 10 недель):")
    print("    t      x_t    x_hat_t    error     S_t[1]")
    for t in range(1, 11):
        print(f"    {t:2d}   {X[t - 1]:6.2f}   {pred[t - 1]:8.2f}   {error[t - 1]:8.2f}   {S[t]:8.2f}")
    print("    ...")
    print("    Таблица результатов (последние 10 недель):")
    for t in range(N - 9, N + 1):
        if t <= N:
            print(f"    {t:2d}   {X[t - 1]:6.2f}   {pred[t - 1]:8.2f}   {error[t - 1]:8.2f}   {S[t]:8.2f}")

    np.savetxt(out_dir / f'constant_alpha_{alpha}_results.csv',
               np.column_stack([weeks, X, pred, error, S[1:]]),
               delimiter=',', header='t,x_t,x_hat_t,error,S_t[1]', comments='')

    # График для постоянной модели
    plt.figure(figsize=(12, 6))
    plt.plot(weeks, X, 'b-o', label='Фактические данные x(t)', markersize=4, linewidth=1.5)
    plt.plot(weeks, pred, 'r--s', label=f'Прогноз x_hat_t, α={alpha}', markersize=4, linewidth=1.5)
    plt.plot(31, future_pred_const, 'm*', markersize=12, label='Прогноз на 31-ю неделю', zorder=5)
    plt.title(f'Постоянная модель (n=0), α={alpha}', fontsize=14)
    plt.xlabel('Неделя (t)', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'02_constant_model_alpha_{alpha}.png', dpi=150)
    plt.show()


#Начальные коэффициенты линейной модели по МНК

print("\n" + "=" * 60)
print("ПУНКТ 3. Начальные коэффициенты линейной модели (МНК)")
print("=" * 60)
print("  Модель: x(t) = A0(0) + A1(0) * t")
print("  Используем метод наименьших квадратов для нахождения A0(0) и A1(0)")

# Метод наименьших квадратов для линейной регрессии
# y = a0 + a1 * x, где x = t (номер недели), y = x(t)
t_weeks = weeks.astype(float)
n_points = N

# Расчет коэффициентов по формулам из Приложения 1
sum_t = np.sum(t_weeks)
sum_x = np.sum(X)
sum_tx = np.sum(t_weeks * X)
sum_t2 = np.sum(t_weeks ** 2)

# a1 = (n*Σtx - Σt*Σx) / (n*Σt² - (Σt)²)
A1_init = (n_points * sum_tx - sum_t * sum_x) / (n_points * sum_t2 - sum_t ** 2)
# a0 = (Σx - a1*Σt) / n
A0_init = (sum_x - A1_init * sum_t) / n_points

print(f"  A0(0) = {A0_init:.4f} (пересечение с осью ординат)")
print(f"  A1(0) = {A1_init:.4f} (тангенс угла наклона)")

# Построение графика аппроксимирующей прямой
plt.figure(figsize=(12, 6))
plt.plot(weeks, X, 'b-o', label='Исходные данные', markersize=4, linewidth=1.5)
plt.plot(weeks, A0_init + A1_init * weeks, 'r--', linewidth=2,
         label=f'Линейная аппроксимация: x = {A0_init:.2f} + {A1_init:.2f}·t')
plt.title('Рис.3. Линейная аппроксимация данных (МНК)', fontsize=14)
plt.xlabel('Неделя (t)', fontsize=12)
plt.ylabel('x(t)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / '03_linear_regression_init.png', dpi=150)
plt.show()
print(f"  График сохранен: {out_dir / '03_linear_regression_init.png'}")



print("Многократное экспоненциальное сглаживание")



smooth_results = {}

for alpha in alpha_values:
    print(f"\n  --- α = {alpha} ---")
    beta_val = 1 - alpha

    # Инициализация сглаженных величин по формулам (7) и (8)
    # S0[1] = A0(0) - (β/α)·A1(0)
    # S0[2] = A0(0) - (2β/α)·A1(0)
    S1 = np.zeros(N + 1)
    S2 = np.zeros(N + 1)

    S1[0] = A0_init - (beta_val / alpha) * A1_init
    S2[0] = A0_init - 2 * (beta_val / alpha) * A1_init

    print(f"Начальные значения:")
    print(f"S0[1] = {S1[0]:.4f}")
    print(f"S0[2] = {S2[0]:.4f}")

    # Рекуррентное сглаживание
    # S_t[1] = α·x_t + β·S_{t-1}[1]
    # S_t[2] = α·S_t[1] + β·S_{t-1}[2]
    for i in range(1, N + 1):
        S1[i] = alpha * X[i - 1] + beta_val * S1[i - 1]
        S2[i] = alpha * S1[i] + beta_val * S2[i - 1]

    smooth_results[alpha] = {'S1': S1, 'S2': S2}

    print(f"    Последние значения:")
    print(f"      S_{N}[1] = {S1[N]:.4f}")
    print(f"      S_{N}[2] = {S2[N]:.4f}")

    # Графики S[1] и S[2]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(weeks, X, 'b-', alpha=0.5, label='Исходные данные', linewidth=1)
    plt.plot(weeks, S1[1:], 'r-', linewidth=2, label='S_t[1]')
    plt.title(f'S_t[1] (α={alpha})', fontsize=12)
    plt.xlabel('Неделя (t)')
    plt.ylabel('Значение')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(weeks, X, 'b-', alpha=0.5, label='Исходные данные', linewidth=1)
    plt.plot(weeks, S2[1:], 'g-', linewidth=2, label='S_t[2]')
    plt.title(f'S_t[2] (α={alpha})', fontsize=12)
    plt.xlabel('Неделя (t)')
    plt.ylabel('Значение')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f'04_smoothing_alpha_{alpha}.png', dpi=150)
    plt.show()

# ПУНКТ 5: Линейная модель (n=1) для прогнозирования (m=1)
print("\n" + "=" * 60)
print("ПУНКТ 5. Линейная модель (n=1) для прогнозирования с m=1")
print("=" * 60)
print("  Модель: x_hat_{t+m} = A0(t) + m·A1(t)")
print("  При m=1: x_hat_{t+1} = A0(t) + A1(t)")

linear_m1_results = {}

for alpha in alpha_values:
    print(f"\n  --- α = {alpha} ---")
    beta_val = 1 - alpha

    # Инициализация сглаженных величин (как в пункте 4)
    S1 = np.zeros(N + 1)
    S2 = np.zeros(N + 1)

    S1[0] = A0_init - (beta_val / alpha) * A1_init
    S2[0] = A0_init - 2 * (beta_val / alpha) * A1_init

    # Массивы для коэффициентов модели
    A0_coef = np.zeros(N + 1)
    A1_coef = np.zeros(N + 1)

    A0_coef[0] = A0_init
    A1_coef[0] = A1_init

    # Прогнозы и ошибки
    pred_m1 = np.zeros(N)
    error_m1 = np.zeros(N)

    # Рекуррентные вычисления
    for i in range(1, N + 1):
        # Прогноз на текущий момент: x_hat_t = A0(t-1) + A1(t-1)
        pred_m1[i - 1] = A0_coef[i - 1] + A1_coef[i - 1]
        error_m1[i - 1] = X[i - 1] - pred_m1[i - 1]

        # Обновление сглаженных величин
        S1[i] = alpha * X[i - 1] + beta_val * S1[i - 1]
        S2[i] = alpha * S1[i] + beta_val * S2[i - 1]

        # Обновление коэффициентов модели по формулам (6)
        # A0(t) = 2·S1(t) - S2(t)
        # A1(t) = (α/β)·(S1(t) - S2(t))
        A0_coef[i] = 2 * S1[i] - S2[i]
        A1_coef[i] = (alpha / beta_val) * (S1[i] - S2[i])

    # Дисперсия ошибки по формуле (10): n = 1, знаменатель = N - 2
    var_error_m1 = np.sum(error_m1 ** 2) / (N - 2)

    # Прогноз на следующую неделю (t = N+1)
    future_pred_m1 = A0_coef[N] + A1_coef[N]

    linear_m1_results[alpha] = {
        'pred': pred_m1, 'error': error_m1,
        'var_error': var_error_m1, 'future_pred': future_pred_m1,
        'A0': A0_coef, 'A1': A1_coef
    }

    print(f"    Дисперсия ошибки: s² = {var_error_m1:.4f}")
    print(f"    Прогноз на 31-ю неделю: x_hat_31 = {future_pred_m1:.2f}")

    print("\n    Таблица результатов (первые 10 недель):")
    print("    t      x_t    x_hat_t    error     A0(t-1)   A1(t-1)")
    for t in range(1, 11):
        print(
            f"    {t:2d}   {X[t - 1]:6.2f}   {pred_m1[t - 1]:8.2f}   {error_m1[t - 1]:8.2f}   {A0_coef[t - 1]:8.2f}   {A1_coef[t - 1]:8.2f}")


    np.savetxt(out_dir / f'linear_m1_alpha_{alpha}_results.csv',
               np.column_stack([weeks, X, pred_m1, error_m1, A0_coef[1:], A1_coef[1:]]),
               delimiter=',', header='t,x_t,x_hat_t,error,A0_t,A1_t', comments='')

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(weeks, X, 'b-o', label='Фактические данные x(t)', markersize=4, linewidth=1.5)
    plt.plot(weeks, pred_m1, 'r--s', label=f'Прогноз x_hat_t, α={alpha}', markersize=4, linewidth=1.5)
    plt.plot(31, future_pred_m1, 'm*', markersize=12, label='Прогноз на 31-ю неделю', zorder=5)
    plt.title(f'Линейная модель (n=1), m=1, α={alpha}', fontsize=14)
    plt.xlabel('Неделя (t)', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'05_linear_m1_alpha_{alpha}.png', dpi=150)
    plt.show()





print("Линейная модель (n=1) для прогнозирования с m=5")

print("  Модель: x_hat_{t+m} = A0(t) + m·A1(t)")
print("  При m=5: x_hat_{t+5} = A0(t) + 5·A1(t)")

linear_m5_results = {}

for alpha in alpha_values:
    print(f"\n  --- α = {alpha} ---")
    beta_val = 1 - alpha

    # Инициализация сглаженных величин
    S1 = np.zeros(N + 1)
    S2 = np.zeros(N + 1)

    S1[0] = A0_init - (beta_val / alpha) * A1_init
    S2[0] = A0_init - 2 * (beta_val / alpha) * A1_init

    # Массивы для коэффициентов модели
    A0_coef = np.zeros(N + 1)
    A1_coef = np.zeros(N + 1)

    A0_coef[0] = A0_init
    A1_coef[0] = A1_init

    # Обновление коэффициентов модели для всех t
    for i in range(1, N + 1):
        S1[i] = alpha * X[i - 1] + beta_val * S1[i - 1]
        S2[i] = alpha * S1[i] + beta_val * S2[i - 1]
        A0_coef[i] = 2 * S1[i] - S2[i]
        A1_coef[i] = (alpha / beta_val) * (S1[i] - S2[i])

    # Прогнозы для m=5: x_hat_{t+5} = A0(t) + 5·A1(t)
    m = 5
    origins = []  # моменты t, для которых делаем прогноз
    targets = []  # моменты t+m, на которые прогнозируем
    pred_m5 = []  # прогнозы
    actual_m5 = []  # фактические значения
    error_m5 = []  # ошибки

    for i in range(N - m):  # i = 0..N-m-1, прогноз из t = i+1 на t+m
        t_origin = i + 1  # момент t
        t_target = t_origin + m  # момент t+m
        pred_val = A0_coef[i] + m * A1_coef[i]
        actual_val = X[i + m]  # X[t+m-1] из-за 0-индексации

        origins.append(t_origin)
        targets.append(t_target)
        pred_m5.append(pred_val)
        actual_m5.append(actual_val)
        error_m5.append(actual_val - pred_val)

    pred_m5 = np.array(pred_m5)
    actual_m5 = np.array(actual_m5)
    error_m5 = np.array(error_m5)
    origins = np.array(origins)
    targets = np.array(targets)

    # Дисперсия ошибки (знаменатель: количество прогнозов - 2)
    n_forecasts = len(error_m5)
    var_error_m5 = np.sum(error_m5 ** 2) / (n_forecasts - 2)

    # Прогноз на будущее: из последней точки (t=N) на t=N+m
    future_pred_m5 = A0_coef[N] + m * A1_coef[N]

    linear_m5_results[alpha] = {
        'pred': pred_m5, 'actual': actual_m5, 'error': error_m5,
        'var_error': var_error_m5, 'future_pred': future_pred_m5,
        'origins': origins, 'targets': targets
    }

    print(f"Количество прогнозов: {n_forecasts}")
    print(f"Дисперсия ошибки: s² = {var_error_m5:.4f}")
    print(f"Прогноз на 35-ю неделю (из t=30, m=5): x_hat_35 = {future_pred_m5:.2f}")

    print("\nаблица результатов (первые 10 прогнозов):")
    print("t (исходная)   t+m (целевая)   x_target   x_hat_target   error")
    for k in range(min(10, n_forecasts)):
        print(
            f"    {origins[k]:4d}          {targets[k]:4d}        {actual_m5[k]:8.2f}   {pred_m5[k]:10.2f}   {error_m5[k]:8.2f}")

    np.savetxt(out_dir / f'linear_m5_alpha_{alpha}_results.csv',
               np.column_stack([origins, targets, actual_m5, pred_m5, error_m5]),
               delimiter=',', header='origin_week,target_week,actual,forecast,error', comments='')

    plt.figure(figsize=(12, 6))
    plt.plot(weeks, X, 'b-o', label='Фактические данные x(t)', markersize=4, linewidth=1.5)
    plt.plot(targets, pred_m5, 'g--^', label=f'Прогноз x_hat_t, m=5, α={alpha}', markersize=4, linewidth=1.5)
    plt.plot(35, future_pred_m5, 'm*', markersize=12, label='Прогноз на 35-ю неделю', zorder=5)
    plt.title(f'Линейная модель (n=1), m=5, α={alpha}', fontsize=14)
    plt.xlabel('Неделя (t)', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'06_linear_m5_alpha_{alpha}.png', dpi=150)
    plt.show()


#Рекомендация по выбору модели

print("Рекомендация по выбору модели")

print("\n  Анализ разностей:")

# Разности первого порядка: Δx(t) = x_{t+1} - x_t
diff1 = np.diff(X)
mean_diff1 = np.mean(diff1)
print(f"Среднее разностей первого порядка: Δx̄ = {mean_diff1:.2f}")

# Разности второго порядка: Δ²x(t) = Δx(t+1) - Δx(t)
diff2 = np.diff(diff1)
mean_diff2 = np.mean(diff2)
print(f"    Среднее разностей второго порядка: Δ²x̄ = {mean_diff2:.2f}")

# Отношение соседних наблюдений для проверки экспоненциального закона
ratios = X[1:] / X[:-1]
mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios, ddof=1)
print(f"    Среднее отношение соседних наблюдений: {mean_ratio:.4f}")
print(f"    Стандартное отклонение отношения: {std_ratio:.4f}")

# Определение порядка полинома по алгоритму из методички
print("\n  Определение порядка полинома:")

# Критерии:
# - Если разности первого порядка колеблются около нуля -> n=0 (постоянная модель)
# - Если среднее разностей первого порядка отлично от нуля, а вторых = 0 -> n=1 (линейная модель)
# - Если наблюдается систематический рост разностей -> возможно экспонента

# Проверка на экспоненциальный рост
exponential_possible = (std_ratio / mean_ratio) < 0.1

# Сравнение средних разностей с нулем с учетом масштаба данных
threshold_1 = 0.01 * np.std(X)  # порог для первых разностей
threshold_2 = 0.1 * np.std(diff1)  # порог для вторых разностей

if abs(mean_diff1) < threshold_1:
    recommended_order = 0
    recommended_model = "постоянная модель (n=0)"
    reason = "разности первого порядка колеблются около нуля"
elif abs(mean_diff2) < threshold_2:
    recommended_order = 1
    recommended_model = "линейная модель (n=1)"
    reason = "среднее первых разностей отлично от нуля, среднее вторых разностей близко к нулю"
else:
    recommended_order = 2
    recommended_model = "квадратичная модель (n=2)"
    reason = "средние разностей значимо отличаются от нуля"

# Дополнительная проверка на экспоненту
if exponential_possible and recommended_order == 0:
    recommended_model += " (возможен экспоненциальный рост)"
    reason += "; отношение соседних наблюдений почти постоянно"

print(f"    Рекомендуемый порядок полинома: n = {recommended_order}")
print(f"    Рекомендуемая модель: {recommended_model}")
print(f"    Обоснование: {reason}")

# Сравнение дисперсий ошибок для эмпирического подтверждения
print("\n  Эмпирическое сравнение моделей по дисперсии ошибки:")

# Для постоянной модели выбираем лучший α
best_const_alpha = 0.1 if const_results[0.1]['var_error'] < const_results[0.3]['var_error'] else 0.3
print(
    f"    Постоянная модель: минимальная дисперсия ошибки = {const_results[best_const_alpha]['var_error']:.4f} (α={best_const_alpha})")

# Для линейной модели (m=1) выбираем лучший α
best_linear_alpha = 0.1 if linear_m1_results[0.1]['var_error'] < linear_m1_results[0.3]['var_error'] else 0.3
print(
    f"    Линейная модель (m=1): минимальная дисперсия ошибки = {linear_m1_results[best_linear_alpha]['var_error']:.4f} (α={best_linear_alpha})")

# Для линейной модели (m=5) выбираем лучший α
best_m5_alpha = 0.1 if linear_m5_results[0.1]['var_error'] < linear_m5_results[0.3]['var_error'] else 0.3
print(
    f"    Линейная модель (m=5): минимальная дисперсия ошибки = {linear_m5_results[best_m5_alpha]['var_error']:.4f} (α={best_m5_alpha})")

# Определение эмпирически лучшей модели
empirical_best_model = ""
if linear_m1_results[best_linear_alpha]['var_error'] < const_results[best_const_alpha]['var_error']:
    empirical_best_model = "линейная модель"
    empirical_comparison = f"Линейная модель дает меньшую дисперсию ошибки ({linear_m1_results[best_linear_alpha]['var_error']:.4f}) чем постоянная ({const_results[best_const_alpha]['var_error']:.4f})"
else:
    empirical_best_model = "постоянная модель"
    empirical_comparison = f"Постоянная модель дает меньшую дисперсию ошибки ({const_results[best_const_alpha]['var_error']:.4f}) чем линейная ({linear_m1_results[best_linear_alpha]['var_error']:.4f})"

print(f"    Эмпирически лучшая модель: {empirical_best_model}")
print(f"    {empirical_comparison}")

# Итоговый вывод
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)

if recommended_order == 0:
    if empirical_best_model == "линейная модель":
        conclusion = f"По анализу разностей рекомендуется {recommended_model}, однако по дисперсии ошибки прогноза лучше работает линейная модель. Рекомендуется использовать линейную модель (n=1) с α={best_linear_alpha} для прогнозирования."
    else:
        conclusion = f"Рекомендуется использовать {recommended_model} с α={best_const_alpha}, так как разности первого порядка колеблются около нуля, и дисперсия ошибки прогноза минимальна."
elif recommended_order == 1:
    if empirical_best_model == "постоянная модель":
        conclusion = f"По анализу разностей рекомендуется {recommended_model}, но постоянная модель показывает меньшую дисперсию ошибки. Возможно, данных недостаточно для уверенного выбора линейной модели."
    else:
        conclusion = f"Рекомендуется использовать {recommended_model} с α={best_linear_alpha}, так как разности первого порядка имеют ненулевое среднее, а разности второго порядка близки к нулю."
else:
    conclusion = f"Рекомендуется использовать {recommended_model}. Если вычислительные ресурсы позволяют, можно попробовать модель более высокого порядка."

print(f"\n  {conclusion}")

# Построение графика разностей
print("\n  Построение графиков разностей...")

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(weeks, X, 'b-o', markersize=4, linewidth=1.5)
plt.title('Исходный временной ряд x(t)', fontsize=12)
plt.ylabel('x(t)')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(weeks[1:], diff1, 'g-s', markersize=4, linewidth=1.5)
plt.axhline(y=mean_diff1, color='r', linestyle='--', label=f'Среднее = {mean_diff1:.2f}')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.title('Разности первого порядка Δx(t)', fontsize=12)
plt.ylabel('Δx(t)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(weeks[2:], diff2, 'm-^', markersize=4, linewidth=1.5)
plt.axhline(y=mean_diff2, color='r', linestyle='--', label=f'Среднее = {mean_diff2:.2f}')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.title('Разности второго порядка Δ²x(t)', fontsize=12)
plt.xlabel('Неделя (t)', fontsize=12)
plt.ylabel('Δ²x(t)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(out_dir / '07_differences_analysis.png', dpi=150)
plt.show()



# Сводная таблица результатов


print("Сводная таблица результатов")


print("\n  Постоянная модель (n=0):")
print(
    f"    α = 0.1: дисперсия ошибки = {const_results[0.1]['var_error']:.4f}, прогноз на 31 неделю = {const_results[0.1]['future_pred']:.2f}")
print(
    f"    α = 0.3: дисперсия ошибки = {const_results[0.3]['var_error']:.4f}, прогноз на 31 неделю = {const_results[0.3]['future_pred']:.2f}")

print("\n  Линейная модель (n=1), m=1:")
print(
    f"    α = 0.1: дисперсия ошибки = {linear_m1_results[0.1]['var_error']:.4f}, прогноз на 31 неделю = {linear_m1_results[0.1]['future_pred']:.2f}")
print(
    f"    α = 0.3: дисперсия ошибки = {linear_m1_results[0.3]['var_error']:.4f}, прогноз на 31 неделю = {linear_m1_results[0.3]['future_pred']:.2f}")

print("\n  Линейная модель (n=1), m=5:")
print(
    f"    α = 0.1: дисперсия ошибки = {linear_m5_results[0.1]['var_error']:.4f}, прогноз на 35 неделю = {linear_m5_results[0.1]['future_pred']:.2f}")
print(
    f"    α = 0.3: дисперсия ошибки = {linear_m5_results[0.3]['var_error']:.4f}, прогноз на 35 неделю = {linear_m5_results[0.3]['future_pred']:.2f}")
