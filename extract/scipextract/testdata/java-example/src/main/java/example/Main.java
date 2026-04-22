package example;

/** Entry point. */
public class Main {
    public static void main(String[] args) {
        int sum = Math.add(1, 2);
        int product = Math.multiply(3, 4);
        System.out.println(sum + " " + product);

        Counter c = new Counter();
        c.bump();
        System.out.println(c.value());

        System.out.println(Math.isEven(42));
    }
}
