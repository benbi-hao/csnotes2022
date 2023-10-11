package leetcode.ds;

import java.util.Deque;
import java.util.ArrayDeque;

public class MyQueue {
    private Deque<Integer> in;
    private Deque<Integer> out;

    public MyQueue() {
        in = new ArrayDeque<>();
        out = new ArrayDeque<>();
    }
    
    public void push(int x) {
        in.push(x);
    }
    
    public int pop() {
        if (out.isEmpty()) {
            in2out();
        }
        return out.pop();
    }
    
    public int peek() {
        if (out.isEmpty()) {
            in2out();
        }
        return out.peek();
    }
    
    public boolean empty() {
        return in.isEmpty() && out.isEmpty();
    }

    private void in2out() {
        while (!in.isEmpty()) {
            out.push(in.pop());
        }
    }
}
