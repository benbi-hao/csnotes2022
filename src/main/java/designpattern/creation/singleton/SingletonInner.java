package designpattern.creation.singleton;

public class SingletonInner {
    private SingletonInner() {}

    public static SingletonInner getInstance() {
        return InnerClass.INSTANCE;
    }

    private static class InnerClass {
        private final static SingletonInner INSTANCE = new SingletonInner();
    }
}
