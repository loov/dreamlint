/** Adds two numbers. */
export function add(a: number, b: number): number {
    return a + b;
}

/** Multiplies two numbers by repeated addition. */
export function multiply(a: number, b: number): number {
    let result = 0;
    for (let i = 0; i < b; i++) {
        result = add(result, a);
    }
    return result;
}
