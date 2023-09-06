package concurrent;

public class OddEven {
    private static volatile Boolean oddFlag = false;

    public static void main(String[] args) {
        new Thread(() -> {
            int number = 1;
            while (number <= 500){
                synchronized(oddFlag) {
                    if (!oddFlag) {
                        System.out.println(number);
                        number += 2;
                        oddFlag = true;
                    }
                }
            }
                
        }).start();

        new Thread(() -> {
            int number = 2;
            while (number <= 500){
                synchronized(oddFlag) {
                    if (oddFlag) {
                        System.out.println(number);
                        number += 2;
                        oddFlag = false;
                    }
                }
            }
                
        }).start();
    }
}
