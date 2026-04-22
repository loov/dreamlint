package example;

/** Math helpers for the SCIP extractor golden test. */
public class Math {
    /** Adds two numbers. */
    public static int add(int a, int b) {
        return a + b;
    }

    /** Multiplies via repeated addition. */
    public static int multiply(int a, int b) {
        int result = 0;
        for (int i = 0; i < b; i++) {
            result = add(result, a);
        }
        return result;
    }

    /** Mutual recursion: delegates to isOdd. */
    public static boolean isEven(int n) {
        if (n == 0) return true;
        return isOdd(n - 1);
    }

    /** Mutual recursion: delegates to isEven. */
    public static boolean isOdd(int n) {
        if (n == 0) return false;
        return isEven(n - 1);
    }
}
