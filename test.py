def fun(x: int, n: int):
    if n == 0:
        return 1
    t = fun(x, n // 2)
    if (n % 2 == 1):
        return t * t * x
    return t * t


print(fun(, 6))
