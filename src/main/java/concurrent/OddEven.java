package concurrent;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import concurrent.util.MySemaphore;

public class OddEven {


    public static void main(String[] args) {
        // 1. 伪自旋
        // rawSpin();

        // 2. CAS + 自旋
        // casSpin();

        // 3. synchronized + flag
        syncFlag();

        // 4. synchronized模拟semaphore
        // semaphoreWithSync();
        // mySemaphore();

        // 5. semaphore
        // rawSemaphore();

        // 6. ReentrantLock + Condition
        // relockFlag();

        // 7. 其他复杂方式可以说都是基于上面的这些方式
    }

    // 1. 伪自旋
    // oddFlag必须加上volatile关键字，否则两个线程里读到的oddFlag可能会不一致，导致同步失效
    // 因为两个线程对oddFlag检查所期待的状态不一样，且一种状态只被一个线程期待，所以可以完成。实际上多数情况下，多线程的for判断有可能在同一时刻对Flag进行判断，导致伪自旋锁失效
    // 例如，让两个线程打印奇数，两个线程打印偶数，共四个线程，这样就无法确保同步了
    // oddFlag用了基本数据类型，没有用原子类，是因为对其的修改语句非常简单，是简单的赋值，所以本质上是原子的，不会出现问题

    private static volatile boolean oddFlag;

    public static void rawSpin() {
        oddFlag = true;
        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                while(oddFlag) {
                    System.out.println(i);
                    i += 2;
                    oddFlag = false;
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                while(!oddFlag) {
                    System.out.println(i);
                    i += 2;
                    oddFlag = true;
                }
            }
        }).start();
    }

    // 2. CAS + 自旋
    // 这里用了原子类，实际上是想要利用原子类内部的CAS操作
    // 除此之外，原子类内部使用了volatile，因此可见性也保证了
    // 基于CAS和自旋保证了临界代码段的互斥性
    // 临界资源的表示flag有3个值，0表示正在被占用，1表示接下来应打印奇数，2表示接下来应打印偶数
    // 进入临界区时CAS设置flag为0，退出临界区时根据奇偶性设置，这样可以保证奇偶同步，即使有多个线程打印奇数，多个线程打印偶数，也不会出错
    public static void casSpin() {
        AtomicInteger flag = new AtomicInteger(1);
        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                while(!flag.compareAndSet(1, 0)) {
                }
                System.out.println(i);
                i += 2;
                flag.set(2);
            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                while(!flag.compareAndSet(2, 0)) {
                }
                System.out.println(i);
                i += 2;
                flag.set(1);
            }
        }).start();
    }

    // 3. synchronized + flag
    // 没有flag不行，因为synchronized只能保证互斥，不能保证同步，而且wait()和notify()只能实现非选择性通知
    // 对flag判断用if可不可以？这个例子中可以，但是对于多个线程打奇数和多个线程打偶数，就不行了，所以这里还是用了while（规范的写法就是while），
    // 本质原因是这里同种线程用的同一个锁，不能选择性通知，不能保证同步，一个偶数线程打印完了有可能会唤醒一个在等待的偶数线程
    // 可否像以下这样写？（相当于只锁对flag的检查和修改，不锁临界代码区）这个例子中可以，但是对于多个线程打奇数和多个线程打偶数，就不行了，因为有可能还没有执行notify就有多个同种线程到达临界区了
    // synchronized (lock) {
    //     while (!oddFlag) {
    //         lock.wait();
    //     }
    // }
    // System.out.println(i);
    // i += 2;
    // synchronized (lock) {
    //     oddFlag = false;
    //     lock.notifyAll();
    // }
    // 应该遵循规范的写法，wait() 临界区 nofity()全部一起写在syncronized代码块里，这样能保证安全
    // 当不止两个线程时，最好用notifyAll而不是notify

    public static void syncFlag() {
        oddFlag = true;
        Object lock = new Object();
        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                synchronized (lock) {
                    while (!oddFlag) {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println(i);
                    i += 2;
                    oddFlag = false;
                    lock.notify();
                }

            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                synchronized (lock) {
                    while (oddFlag) {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    System.out.println(i);
                    i += 2;
                    oddFlag = true;
                    lock.notify();
                }
            }
        }).start();
        
    }

    // 4. synchronized模拟semaphore
    // 用synchronized机制和两个锁两个count模拟了信号量的操作
    // 因为用synchronized对信号量操作做了互斥，所以效率不如native的信号量高
    // 经测试即使是多个奇数线程和多个偶数线程也正确
    // 可以将PV逻辑封装起来，具体见项目里的concurrent.util.MySemaphore
    // 这里count不加volatile也不影响，为了规范点还是加上了，为什么不加也不影响？这个问题有待深究
    private static volatile int oddCount;
    private static volatile int evenCount;
    public static void semaphoreWithSync() {
        oddCount = 1;
        evenCount = 0;
        Object oddLock = new Object();
        Object evenLock = new Object();

        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                synchronized (oddLock) {
                    oddCount--;
                    if (oddCount < 0) {
                        try {
                            oddLock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
                System.out.println(i);
                i += 2;
                synchronized (evenLock) {
                    evenCount++;
                    if (evenCount <= 0) {
                        evenLock.notify();
                    }
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                synchronized (evenLock) {
                    evenCount--;
                    if (evenCount < 0) {
                        try {
                            evenLock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
                System.out.println(i);
                i += 2;
                synchronized (oddLock) {
                    oddCount++;
                    if (oddCount <= 0) {
                        oddLock.notify();
                    }
                }
            }
        }).start();

    }

    public static void mySemaphore() {
        MySemaphore oddSemaphore = new MySemaphore(1);
        MySemaphore evenSemaphore = new MySemaphore(0);

        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                oddSemaphore.P();
                System.out.println(i);
                i += 2;
                evenSemaphore.V();
            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                evenSemaphore.P();
                System.out.println(i);
                i += 2;
                oddSemaphore.V();
            }
        }).start();
    }

    // 5. semaphore
    // JDK的信号量
    public static void rawSemaphore() {
        Semaphore oddSemaphore = new Semaphore(1);
        Semaphore evenSemaphore = new Semaphore(0);

        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                try {
                    oddSemaphore.acquire();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                
                System.out.println(i);
                i += 2;

                evenSemaphore.release();
            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                try {
                    evenSemaphore.acquire();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                
                System.out.println(i);
                i += 2;

                oddSemaphore.release();
            }
        }).start();
    }

    // 6. ReentrantLock + Condition + flag
    // 总体结构和sychronized + flag差不多，原理也差不多
    // 意外的是，即使使用了Condition::await和signal达到条件等待唤醒，在await外面依旧要使用while结构而非if结构，
    // 否则当多奇线程和多偶线程时，依旧会出错，这个现象和之前在synchronized + flag方法中分析的不符
    // 目前在网上找的解释是虚假唤醒，但是我不确定是不是我实验现象的问题所在，有待深究
    // 总之，按照JDK文档的推荐，await()方法用while包裹才是安全妥当的做法
    public static void relockFlag() {
        oddFlag = true;
        ReentrantLock lock = new ReentrantLock();
        Condition oddCondition = lock.newCondition();
        Condition evenCondition = lock.newCondition();

        new Thread(() -> {
            for (int i = 1; i <= 100;) {
                lock.lock();
                try {
                    while (!oddFlag) {
                        oddCondition.await();
                    }
                    System.out.println(i);
                    i += 2;
                    oddFlag = false;
                    evenCondition.signal();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    lock.unlock();
                }
            }
        }).start();

        new Thread(() -> {
            for (int i = 2; i <= 100;) {
                lock.lock();
                try {
                    while (oddFlag) {
                        evenCondition.await();
                    }
                    System.out.println(i);
                    i += 2;
                    oddFlag = true;
                    oddCondition.signal();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    lock.unlock();
                }
            }
        }).start();
    }

}
