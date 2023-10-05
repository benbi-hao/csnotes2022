package concurrent.util;

/**
 * 基于java sychronized关键字实现的简单信号量（不公平、synchronized互斥调用效率不高）
 */
public class MySemaphore {
    private volatile int count;

    public MySemaphore(int count) {
        this.count = count;
    }

    public synchronized void P() {
        count--;
        if (count < 0) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public synchronized void V() {
        count++;
        if (count <= 0) {
            this.notify();
        }
    }
}
