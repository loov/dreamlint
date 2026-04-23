"""Math helpers for the SCIP extractor golden test."""


def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies via repeated addition."""
    result = 0
    for _ in range(b):
        result = add(result, a)
    return result


class Counter:
    """Counter accumulates a running total."""

    def __init__(self) -> None:
        """Creates a zeroed counter."""
        self.n: int = 0

    def bump(self) -> int:
        """Increments by one, returns the new value."""
        self.n = add(self.n, 1)
        return self.n

    def value(self) -> int:
        """Returns the current value."""
        return self.n


def is_even(n: int) -> bool:
    """Mutual recursion: delegates to is_odd."""
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n: int) -> bool:
    """Mutual recursion: delegates to is_even."""
    if n == 0:
        return False
    return is_even(n - 1)
