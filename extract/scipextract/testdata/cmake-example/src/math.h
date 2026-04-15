#pragma once

namespace math {

// Add returns the sum of a and b.
int add(int a, int b);

// Multiply returns the product of a and b, built by repeated addition.
int multiply(int a, int b);

// Counter accumulates bumps into a running total.
class Counter {
public:
    Counter();
    // bump increments the counter and returns the new value.
    int bump();
    // value returns the current counter.
    int value() const;

private:
    int n;
};

// is_even reports whether n is even via mutual recursion with is_odd.
bool is_even(int n);

// is_odd reports whether n is odd via mutual recursion with is_even.
bool is_odd(int n);

} // namespace math
