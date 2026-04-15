import { add, Counter, isEven, multiply } from "./math";

function main(): void {
    const sum = add(2, 3);
    const product = multiply(sum, 4);

    const counter = new Counter();
    counter.bump();
    counter.bump();

    console.log(product, counter.value(), isEven(product));
}

main();
