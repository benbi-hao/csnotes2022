package leetcode.ds;

import java.util.ArrayList;
import java.util.List;

public class MinStack {
    List<Integer> array;
    int min;

    public MinStack() {
        array = new ArrayList<>();
        min = Integer.MAX_VALUE;
    }
    
    public void push(int val) {
        array.add(val);
        if (val < min) min = val;
    }
    
    public void pop() {
        int ret = array.remove(array.size() - 1);
        if (ret == min) {
            min = Integer.MAX_VALUE;
            int len = array.size();
            for (int i = 0; i < len; i++) {
                min = Math.min(min, array.get(i));
            }
        }
        
    }
    
    public int top() {
        return array.get(array.size() - 1);
    }
    
    public int getMin() {
        return min;
    }
}
