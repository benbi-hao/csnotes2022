package concurrent.util;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * 基于CAS实现的简单自旋锁（这里用原子类只是为了利用CAS操作）（非公平，一线程独占，无法防止软硬中断，无法应对异常退出）
 */
public class MySpinLock {
    private AtomicInteger atomicInteger;

    public MySpinLock() {
        atomicInteger = new AtomicInteger(1);
    }

    public void lock() {
        while (!atomicInteger.compareAndSet(1, 0)) {
        }
    }

    public void unlock() {
        atomicInteger.set(1);
    }
}
