package designpattern.creation.singleton;

public class SingletonLazy {
    private SingletonLazy() {}

    private static SingletonLazy instance;

    public static SingletonLazy getInstance() {
        if (instance == null) {
            instance = new SingletonLazy();
        }
        return instance;
    }
}
