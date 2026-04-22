package example;

/** Counter accumulates a running total. */
public class Counter {
    private int n;

    /** Creates a zeroed counter. */
    public Counter() {
        this.n = 0;
    }

    /** Increments by one, returns the new value. */
    public int bump() {
        this.n = Math.add(this.n, 1);
        return this.n;
    }

    /** Returns the current value. */
    public int value() {
        return this.n;
    }
}
