import { add, multiply } from "./math";

function main(): void {
    const sum = add(2, 3);
    const product = multiply(sum, 4);
    console.log(product);
}

main();
