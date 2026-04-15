#include "math.h"

namespace math {

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    int result = 0;
    for (int i = 0; i < b; ++i) {
        result = add(result, a);
    }
    return result;
}

Counter::Counter() : n(0) {}

int Counter::bump() {
    n = add(n, 1);
    return n;
}

int Counter::value() const {
    return n;
}

bool is_even(int n) {
    if (n == 0) {
        return true;
    }
    return is_odd(n - 1);
}

bool is_odd(int n) {
    if (n == 0) {
        return false;
    }
    return is_even(n - 1);
}

} // namespace math
