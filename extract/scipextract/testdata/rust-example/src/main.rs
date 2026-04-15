use rust_example::{add, multiply};

fn main() {
    let sum = add(2, 3);
    let product = multiply(sum, 4);
    println!("{}", product);
}
