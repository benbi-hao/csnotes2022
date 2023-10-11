package leetcode.ds;

import java.util.ArrayDeque;
import java.util.Queue;

// public class MyStack {
//     private Queue<Integer> queue1;
//     private Queue<Integer> queue2;


//     public MyStack() {
//         queue1 = new ArrayDeque<>();
//         queue2 = new ArrayDeque<>();
//     }
    
//     public void push(int x) {
//         queue2.offer(x);
//         while (!queue1.isEmpty()) {
//             queue2.offer(queue1.poll());
//         }
//         Queue<Integer> temp = queue1;
//         queue1 = queue2;
//         queue2 = temp;
//     }
    
//     public int pop() {
//         return queue1.poll();
//     }
    
//     public int top() {
//         return queue1.peek();
//     }
    
//     public boolean empty() {
//         return queue1.isEmpty();
//     }
// }

// 1个队列写法
// public class MyStack {
//     private Queue<Integer> queue;


//     public MyStack() {
//         queue = new ArrayDeque<>();
//     }
    
//     public void push(int x) {
//         int cnt = queue.size();
//         queue.offer(x);
//         while (cnt-- > 0) {
//             queue.offer(queue.poll());
//         }
//     }
    
//     public int pop() {
//         return queue.poll();
//     }
    
//     public int top() {
//         return queue.peek();
//     }
    
//     public boolean empty() {
//         return queue.isEmpty();
//     }
// }
