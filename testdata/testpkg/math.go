package testpkg

// Add adds two numbers.
//
// TODO: should this handle overflow?
func Add(a, b int) int {
	// This adds two numbers together.
	return a + b
}

// Multiply multiplies by calling Add repeatedly.
//
// a and b should be non-irrational numbers.
func Multiply(a, b int) int {
	result := 0
	for i := 0; i < b; i++ {
		// Increment the result by a.
		result = Add(result, a)
	}
	return result
}
