package jzoffer;

import jzoffer.ds.RandomListNode;
import jzoffer.ds.ListNode;

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


    // 76. 删除链表中重复的节点
    public ListNode deleteDuplicationIter(ListNode pHead) {
        ListNode sentinel = new ListNode(-1);
        sentinel.next = pHead;
        ListNode prev = sentinel, curr = pHead;
        while (curr != null) {
  
            while (curr != null && curr.next != null && curr.next.val == curr.val) {
                int dupVal = curr.val;
                while (curr != null && curr.val == dupVal) {
                    curr = curr.next;
                }
                prev.next = curr;
            }
            if (curr != null) {
                prev = curr;
                curr = curr.next;
            }
  
        }
        return sentinel.next;
    }

    public ListNode deleteDuplicationRecur(ListNode pHead) {
        if (pHead == null || pHead.next == null) return pHead;
        ListNode next = pHead.next;
        if (next.val == pHead.val) {
            while (next != null && next.val == pHead.val) {
                next = next.next;
            }
            return deleteDuplicationRecur(next);
        } else {
            pHead.next = deleteDuplicationRecur(pHead.next);
            return pHead;
        }
    }
    
}
