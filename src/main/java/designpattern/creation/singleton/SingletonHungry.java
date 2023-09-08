package designpattern.creation.singleton;

public class SingletonHungry {
    private SingletonHungry() {}

    private final static SingletonHungry instance = new SingletonHungry();

    public static SingletonHungry getInstance() {
        return instance;
    }
}
