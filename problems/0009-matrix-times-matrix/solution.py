def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]] | int:
    if not a or not b or len(a[0]) != len(b):
        return -1

    b_cols = list(zip(*b))

    c = [
        [sum(x * y for x, y in zip(row_a, col_b)) for col_b in b_cols]
        for row_a in a
    ]
    return c
