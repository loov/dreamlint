#include "math.h"

#include <cstdio>

int main() {
    int sum = math::add(2, 3);
    int product = math::multiply(sum, 4);
    std::printf("%d\n", product);
    return 0;
}
