import math
import numpy as np
from scipy.optimize import bisect, newton, root_scalar, fixed_point

# f(x) = cos x - x
def f(x):
    return math.cos(x) - x

# Производная
def df(x):
    return -math.sin(x) - 1

# Фи для простой итерации: x = cos(x)
def phi(x):
    return math.cos(x)

# Производная фи (для проверки сходимости)
def dphi(x):
    return -math.sin(x)

# Табулирование
print('Табулирование функции')
for x in np.arange(0, 1.0001, 0.1):
    print(x, '|', f(x))

# Ищем отрезок где f(a)*f(b) < 0
a = 0
b = 1
if f(a) * f(b) > 0:
    print('На отрезке [0; 1] корня нет')
else:
    print('Отрезок с корнем:', a, b)

print()

# Проверка сходимости простой итерации
print("Проверка |phi'(x)| < 1 на отрезке:")
xx = a
while xx <= b:
    print(xx, "|", abs(dphi(xx)))
    xx = xx + 0.01

print()

# Результаты из Excel
excel = {
    ('Бисекция', 0.001): (0.7392578125, 9),
    ('Бисекция', 1e-05): (0.7390823364257812, 16),
    ('Хорды', 0.001): (0.7390781308800257, 3),
    ('Хорды', 1e-05): (0.7390847824489231, 4),
    ('Ньютон x0=0', 0.001): (0.739085133385284, 3),
    ('Ньютон x0=1', 0.001): (0.739085133385284, 2),
    ('Ньютон x0=0', 1e-05): (0.7390851332151607, 4),
    ('Простая итерация x0=0 (excel)', 0.001): (0.7387603198742113, 17),
    ('Простая итерация x0=0 (excel)', 1e-05): (0.7390822985224024, 29),
}
results = []

# Два значения точности (eps)
for eps in [0.001, 0.00001]:
    print('eps =', eps, '')

    # Бисекция
    print('Метод бисекции')
    A = a
    B = b
    i = 0
    while (B - A) / 2 > eps:
        c = (A + B) / 2
        print(i, '| a =', A, '| b =', B, '| c =', c, '| f(c) =', f(c), '| err =', (B - A) / 2)
        if f(A) * f(c) < 0:
            B = c
        else:
            A = c
        i = i + 1
    x1 = (A + B) / 2
    print('Корень =', x1)
    print('Итераций =', i)
    print('Невязка =', abs(f(x1)))
    results.append(('Бисекция', eps, x1, i, abs(f(x1))))

    print()

    # Метод Ньютона для начального приближения x0 = 0
    print('Метод Ньютона, x0 = 0')
    x = 0
    i = 0
    while True:
        x_new = x - f(x) / df(x)
        print(i, '| x =', x, '| x_new =', x_new, '| f(x_new) =', f(x_new), '| err =', abs(x_new - x))
        i = i + 1
        if abs(x_new - x) < eps:
            break
        x = x_new
    x2 = x_new
    print('Корень =', x2)
    print('Итераций =', i-1)
    print('Невязка =', abs(f(x2)))
    results.append(('Ньютон x0=0', eps, x2, i - 1, abs(f(x2))))

    print()

    # Метод Ньютона для начального приближения x0 = 1
    print('Метод Ньютона, x0 = 1')
    x = 1
    i = 0
    while True:
        x_new = x - f(x) / df(x)
        print(i, '| x =', x, '| x_new =', x_new, '| f(x_new) =', f(x_new), '| err =', abs(x_new - x))
        i = i + 1
        if abs(x_new - x) < eps:
            break
        x = x_new
    x3 = x_new
    print('Корень =', x3)
    print('Итераций =', i-1)
    print('Невязка =', abs(f(x3)))
    results.append(('Ньютон x0=1', eps, x3, i - 1, abs(f(x3))))

    print()

    # Метод хорд
    print('Метод хорд')
    A = a
    B = b
    i = 0
    x_old = A
    while True:
        x_new = A - f(A) * (B - A) / (f(B) - f(A))
        print(i, '| a =', A, '| b =', B, '| x =', x_new, '| f(x) =', f(x_new), '| err =', abs(x_new - x_old))
        i = i + 1
        if abs(x_new - x_old) < eps:
            break
        if f(A) * f(x_new) < 0:
            B = x_new
        else:
            A = x_new
        x_old = x_new
    print('Корень =', x_new)
    print('Итераций =', i-1)
    print('Невязка =', abs(f(x_new)))
    results.append(('Хорды', eps, x_new, i - 1, abs(f(x_new))))

    print()

    # Простая итерация
    print('Метод простой итерации')
    x = 0
    i = 0
    while True:
        x_new = phi(x)
        print(i, '| x =', x, '| x_new =', x_new, '| f(x_new) =', f(x_new), '| err =', abs(x_new - x))
        i = i + 1
        if abs(x_new - x) < eps:
            break
        x = x_new
    print('Корень =', x_new)
    print('Итераций =', i-1)
    print('Невязка =', abs(f(x_new)))
    results.append(('Простая итерация x0=0 (код)', eps, x_new, i - 1, abs(f(x_new))))

    print()

    print('Проверка через scipy')
    bisect_val = bisect(f, a, b, xtol=eps)
    print('bisect =', bisect_val)
    newton0_val = newton(f, 0, fprime=df, tol=eps)
    print('newton x0=0 =', newton0_val)
    newton1_val = newton(f, 1, fprime=df, tol=eps)
    print('newton x0=1 =', newton1_val)
    root_scalar_val = root_scalar(f, bracket=[a, b], method='brentq', xtol=eps).root
    print('root_scalar =', root_scalar_val)
    fixed_point_val = fixed_point(phi, 0, xtol=eps)
    print('fixed_point x0=0 =', fixed_point_val)

    scipy_by_method = {
        'Бисекция': bisect_val,
        'Ньютон x0=0': newton0_val,
        'Ньютон x0=1': newton1_val,
        'Хорды': root_scalar_val,
        'Простая итерация x0=0 (код)': fixed_point_val,
    }
    for row in results:
        if row[1] == eps and len(row) == 5:
            name = row[0]
            results[results.index(row)] = row + (scipy_by_method[name],)

    print()


print('Таблица сравнения: своя программа / SciPy / Excel')
header = (f"{'Метод':<32}{'eps':>8}{'Свой корень':>15}{'Итер.':>7}"
          f"{'Невязка':>12}{'SciPy':>15}{'Excel':>15}{'Итер. Excel':>13}")
print('-' * len(header))
print(header)
print('-' * len(header))
for name, eps, root, it, resid, sp in results:
    excel_key = name if name != 'Простая итерация x0=0 (код)' else 'Простая итерация x0=0 (excel)'
    if (excel_key, eps) in excel:
        ex_root, ex_it = excel[(excel_key, eps)]
        ex_root_txt = f'{ex_root:>15.9f}'
        ex_it_txt = f'{ex_it:>11d}'
    else:
        ex_root_txt = f"{'-':>15}"
        ex_it_txt = f"{'-':>11}"
    print(f'{name:<32}{eps:>8.0e}{root:>15.9f}{it:>7d}{resid:>12.2e}'
          f'{sp:>15.9f}{ex_root_txt}{ex_it_txt}')

