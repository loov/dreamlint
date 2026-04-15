#include "math.h"

#include <cstdio>

int main() {
    int sum = math::add(2, 3);
    int product = math::multiply(sum, 4);

    math::Counter counter;
    counter.bump();
    counter.bump();

    std::printf("%d %d even=%d\n", product, counter.value(), math::is_even(product));
    return 0;
}
