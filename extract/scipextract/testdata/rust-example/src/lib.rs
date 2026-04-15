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

/// Counter accumulates bumps into a running total.
pub struct Counter {
    n: i32,
}

impl Counter {
    /// Creates a zeroed Counter.
    pub fn new() -> Self {
        Self { n: 0 }
    }

    /// Increments the counter by one and returns the new value.
    pub fn bump(&mut self) -> i32 {
        self.n = add(self.n, 1);
        self.n
    }

    /// Returns the current counter value.
    pub fn value(&self) -> i32 {
        self.n
    }
}

impl Default for Counter {
    fn default() -> Self {
        Counter::new()
    }
}

/// Mutual recursion: is_even delegates to is_odd.
pub fn is_even(n: i32) -> bool {
    if n == 0 {
        true
    } else {
        is_odd(n - 1)
    }
}

/// Mutual recursion: is_odd delegates to is_even.
pub fn is_odd(n: i32) -> bool {
    if n == 0 {
        false
    } else {
        is_even(n - 1)
    }
}
