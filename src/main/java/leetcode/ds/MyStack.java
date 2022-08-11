package leetcode.ds;

import java.util.LinkedList;
import java.util.Queue;

public class MyStack {
    private Queue<Integer> queueA;
    private Queue<Integer> queueB;
    private Queue<Integer> major;
    private Queue<Integer> minor;


    public MyStack() {
        queueA = new LinkedList<>();
        queueB = new LinkedList<>();
        major = queueA;
        minor = queueB;
    }
    
    public void push(int x) {
        major.offer(x);
    }
    
    public int pop() {
        while(major.size() > 1) {
            minor.offer(major.poll());
        }
        int ret = major.poll();
        interchange();
        return ret;
    }
    
    public int top() {
        while(major.size() > 1) {
            minor.offer(major.poll());
        }
        int ret = major.poll();
        minor.offer(ret);
        interchange();
        return ret;
    }
    
    public boolean empty() {
        return major.isEmpty();
    }

    private void interchange() {
        Queue<Integer> temp = major;
        major = minor;
        minor = temp;
    }
}
