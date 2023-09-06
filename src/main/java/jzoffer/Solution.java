package jzoffer;

import java.util.*;

public class Solution {
    // 59. 滑动窗口的最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> deque = new ArrayDeque<>();
        int n = nums.length;
        int[] ret = new int[n - k + 1];
        int index = 0;
        for (int i = 0; i < n; i++) {
            while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            if (deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }

            if (i >= k - 1) {
                ret[index++] = nums[deque.peekFirst()];
            }
        }
        return ret;
    }

    
}
