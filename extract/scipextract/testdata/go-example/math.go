package example

// Add returns the sum of a and b.
func Add(a, b int) int {
	return a + b
}

// Multiply returns a * b via repeated addition.
func Multiply(a, b int) int {
	result := 0
	for range b {
		result = Add(result, a)
	}
	return result
}

// Counter accumulates a running total.
type Counter struct {
	n int
}

// Bump increments the counter by one and returns the new value.
func (c *Counter) Bump() int {
	c.n = Add(c.n, 1)
	return c.n
}

// Value returns the current counter value.
func (c Counter) Value() int {
	return c.n
}

// IsEven reports whether n is even (mutual recursion with IsOdd).
func IsEven(n int) bool {
	if n == 0 {
		return true
	}
	return IsOdd(n - 1)
}

// IsOdd reports whether n is odd (mutual recursion with IsEven).
func IsOdd(n int) bool {
	if n == 0 {
		return false
	}
	return IsEven(n - 1)
}
