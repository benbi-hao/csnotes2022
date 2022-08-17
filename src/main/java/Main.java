public class Main {
    public static void main(String[] args){
        Father x = new Son();
        x.eat();
        
    }
}

class Father implements Eatable {
    public void eat() {
        System.out.println("Father eat");
    }
}

class Son extends Father {
    public void eat() {
        System.out.println("Son Eat.");
    }
}

interface Eatable {
    void eat();
}
