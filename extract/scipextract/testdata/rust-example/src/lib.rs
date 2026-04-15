//! Tiny math helpers used by the SCIP extractor golden test.

/// Adds two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiplies two numbers by repeated addition.
pub fn multiply(a: i32, b: i32) -> i32 {
    let mut result = 0;
    for _ in 0..b {
        result = add(result, a);
    }
    result
}
