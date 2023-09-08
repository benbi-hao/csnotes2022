package designpattern.creation.singleton;

public class SingletonDCL {
    private SingletonDCL() {}

    private volatile static SingletonDCL instance;

    public SingletonDCL getInstance() {
        if (instance == null) {
            synchronized(SingletonDCL.class) {
                if (instance == null) {
                    instance = new SingletonDCL();
                }
            }
        }

        return instance;
    }

}