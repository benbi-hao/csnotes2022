package leetcode.ds;

import java.util.Stack;

public class MyQueue {
    private Stack<Integer> storage;
    private Stack<Integer> cache;

    public MyQueue() {
        storage = new Stack<>();
        cache = new Stack<>();
    }
    
    public void push(int x) {
        storage.push(x);
    }
    
    public int pop() {
        while(!storage.isEmpty()) {
            cache.push(storage.pop());
        }
        int ret = cache.pop();
        while(!cache.isEmpty()) {
            storage.push(cache.pop());
        }
        return ret;
    }
    
    public int peek() {
        while(!storage.isEmpty()) {
            cache.push(storage.pop());
        }
        int ret = cache.peek();
        while(!cache.isEmpty()) {
            storage.push(cache.pop());
        }
        return ret;
    }
    
    public boolean empty() {
        return storage.isEmpty();
    }
}
