package jzoffer;

import jzoffer.ds.RandomListNode;
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

    // 35. 复杂链表的复制
    private Map<RandomListNode, RandomListNode> cache;

    public RandomListNode Clone(RandomListNode pHead) {
        cache = new HashMap<>();
        RandomListNode newHead = sequentialClone(pHead);
        RandomListNode pOld = pHead, pNew = newHead;
        while (pOld != null) {
            if (pOld.random == null) {
                pNew.random = null;
            } else {
                pNew.random = cache.get(pOld.random);
            }
            pOld = pOld.next;
            pNew = pNew.next;
        }
        return newHead;

    }

    public RandomListNode sequentialClone(RandomListNode pHead) {
        if (pHead == null) return null;
        RandomListNode cloned = new RandomListNode(pHead.label);
        cloned.next = sequentialClone(pHead.next);
        cache.put(pHead, cloned);
        return cloned;
    }
    
}
