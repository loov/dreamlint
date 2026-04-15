use rust_example::{add, is_even, multiply, Counter};

fn main() {
    let sum = add(2, 3);
    let product = multiply(sum, 4);

    let mut counter = Counter::new();
    counter.bump();
    counter.bump();

    println!("{} {} even={}", product, counter.value(), is_even(product));
}
