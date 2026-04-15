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

/** Counter accumulates bumps into a running total. */
export class Counter {
    private n: number;

    constructor() {
        this.n = 0;
    }

    /** Increments the counter and returns the new value. */
    bump(): number {
        this.n = add(this.n, 1);
        return this.n;
    }

    /** Returns the current counter value. */
    value(): number {
        return this.n;
    }
}

/** Mutual recursion: isEven delegates to isOdd. */
export function isEven(n: number): boolean {
    if (n === 0) {
        return true;
    }
    return isOdd(n - 1);
}

/** Mutual recursion: isOdd delegates to isEven. */
export function isOdd(n: number): boolean {
    if (n === 0) {
        return false;
    }
    return isEven(n - 1);
}
