"""Entry point."""

from math_helpers import Counter, add, is_even, multiply


def main() -> None:
    """Exercises the math helpers."""
    total = add(1, 2)
    product = multiply(3, 4)
    print(total, product)

    c = Counter()
    c.bump()
    print(c.value())

    print(is_even(42))
