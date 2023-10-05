package concurrent;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class ABC {
    public static void main(String[] args) {
        // 伪自旋
        rawSpin();

        // CAS + 自旋
        casSpin();

        // synchronized + flag
        syncFlag();

        // 信号量
        rawSemaphore();

        // ReentrantLock + Condition + flag
        relockFlag();
    }

    private static volatile int flag;

    // 伪自旋
    public static void rawSpin() {
        flag = 1;
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                while (flag == 1) {
                    System.out.println('A');
                    i++;
                    flag = 2;
                }
            }
        }).start();
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                while (flag == 2) {
                    System.out.println('B');
                    i++;
                    flag = 3;
                }
            }
        }).start();
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                while (flag == 3) {
                    System.out.println('C');
                    i++;
                    flag = 1;
                }
            }
        }).start();
    }

    // CAS + 自旋
    public static void casSpin() {
        AtomicInteger flag = new AtomicInteger(1);
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                while (!flag.compareAndSet(1, 0)){}
                System.out.println('A');
                i++;
                flag.set(2);
            }
        }).start();
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                while (!flag.compareAndSet(2, 0)){}
                System.out.println('B');
                i++;
                flag.set(3);
            }
        }).start();
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                while (!flag.compareAndSet(3, 0)){}
                System.out.println('C');
                i++;
                flag.set(1);
            }
        }).start();
    }

    // synchronized + flag
    // 必须notifyAll，因为不一定会唤醒正确的下一个线程
    public static void syncFlag() {
        flag = 1;
        Object lock = new Object();
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                synchronized (lock) {
                    while (flag != 1) {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println('A');
                    i++;
                    flag = 2;
                    lock.notifyAll();
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                synchronized (lock) {
                    while (flag != 2) {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println('B');
                    i++;
                    flag = 3;
                    lock.notifyAll();
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                synchronized (lock) {
                    while (flag != 3) {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println('C');
                    i++;
                    flag = 1;
                    lock.notifyAll();
                }
            }
        }).start();
    }

    // 信号量
    public static void rawSemaphore() {
        Semaphore semaphoreA = new Semaphore(1);
        Semaphore semaphoreB = new Semaphore(0);
        Semaphore semaphoreC = new Semaphore(0);        
        
        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                try {
                    semaphoreA.acquire();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println('A');
                i++;
                semaphoreB.release();
            }
        }).start();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                try {
                    semaphoreB.acquire();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println('B');
                i++;
                semaphoreC.release();
            }
        }).start();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                try {
                    semaphoreC.acquire();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println('C');
                i++;
                semaphoreA.release();
            }
        }).start();
    }

    // ReentrantLock + Condition + flag
    // 这里用了signal，而不用signalAll，是因为唤醒的线程一定是对的线程
    public static void relockFlag() {
        flag = 1;
        ReentrantLock lock = new ReentrantLock();
        Condition conditionA = lock.newCondition();
        Condition conditionB = lock.newCondition();
        Condition conditionC = lock.newCondition();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                lock.lock();
                try {
                    while (flag != 1) {
                        conditionA.await();
                    }
                    System.out.println('A');
                    i++;
                    flag = 2;
                    conditionB.signal();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    lock.unlock();
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                lock.lock();
                try {
                    while (flag != 2) {
                        conditionB.await();
                    }
                    System.out.println('B');
                    i++;
                    flag = 3;
                    conditionC.signal();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    lock.unlock();
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 1; i <= 30;) {
                lock.lock();
                try {
                    while (flag != 3) {
                        conditionC.await();
                    }
                    System.out.println('C');
                    i++;
                    flag = 1;
                    conditionA.signal();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    lock.unlock();
                }
            }
        }).start();

        
    }
}
