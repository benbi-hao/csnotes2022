package leetcode;

import leetcode.ds.ListNode;

public class Solution {
    /**
     * 链表
     */

     // 160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA;
        ListNode b = headB;
        while (a != b) {
            a = a != null ? a.next : headB;
            b = b != null ? b.next : headA;
        }

        return a;
    }

    // 206. 反转链表
    public ListNode reverseList(ListNode head){             // 迭代法
        ListNode prev = null, next = null;
        ListNode curr = head;
        while (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    public ListNode reverseListRecur(ListNode head) {       // 递归法
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverseListRecur(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    public ListNode reverseListInsert(ListNode head) {      // 头插法
        ListNode sentinel = new ListNode(0);
        ListNode curr = head, next = null;
        while(curr != null) {
            next = curr.next;
            curr.next = sentinel.next;
            sentinel.next = curr;
            curr = next;
        }
        return sentinel.next;
    }

    // 21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode sentinel = new ListNode(0);
        ListNode p = sentinel;
        while (list1 != null && list2 != null) {
            if (list1.val > list2.val) {
                p.next = list2;
                list2 = list2.next;
            } else {
                p.next = list1;
                list1 = list1.next;
            }
            p = p.next;
        }
        if (list1 != null) {
            p.next = list1;
        } else {
            p.next = list2;
        }
        return sentinel.next;
    }

    // 83. 删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {       // 迭代法
        if (head == null) return null;
        ListNode curr = head;
        while (curr.next != null) {
            if (curr.val == curr.next.val) {
                curr.next = curr.next.next;
            }else {
                curr = curr.next;
            }
        }
        return head;
    }

    public ListNode deleteDuplicatesRecur(ListNode head) {  // 递归法
        if (head == null || head.next == null) { return head; }
        head.next = deleteDuplicatesRecur(head.next);
        return head.val == head.next.val ? head.next : head;
    }

    // 19. 删除链表的倒数第N个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode sentinel = new ListNode(0, head);
        ListNode first = sentinel, second = null;
        while (n > 0) {
            first = first.next;
            n--;
        }
        second = sentinel;
        while (first.next != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return sentinel.next;
    }

    // 24. 两两交换链表中的结点
    public ListNode swapPairs(ListNode head) {          // 递归法
        if (head == null || head.next == null) return head;
        head.next.next = swapPairs(head.next.next);
        ListNode ret = head.next;
        head.next = ret.next;
        ret.next = head;
        return ret;
    }

    public ListNode swapPairsIter(ListNode head) {      // 迭代法
        if (head == null || head.next == null) return head;
        ListNode ret = head.next;
        ListNode curr = head, next = null;
        while (curr != null && curr.next != null) {
            next = curr.next.next;
            curr.next.next = curr;
            if (next == null || next.next == null) {
                curr.next = next;
            } else {
                curr.next = next.next;
            }
            curr = next;
        }
        return ret;
    }

    // 445. 两数相加2
    // 不反转链表可以用栈来反过来取加数，但是话说回来，既然我都用栈了为什么不新建个链表反转呢
    private int carry;
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {   // 不反转链表
        this.carry = 0;
        ListNode p1 = l1, p2 = l2;
        int len1 = 0, len2 = 0;
        while (p1 != null) {
            p1 = p1.next;
            len1++;
        }
        while (p2 != null) {
            p2 = p2.next;
            len2++;
        }

        int diff = len1 > len2 ? len1 - len2 : len2 - len1;
        ListNode prime = len1 > len2 ? l1 : l2;
        ListNode zerosSentinel = new ListNode(0);
        ListNode pz = zerosSentinel;
        while(diff > 0) {
            pz.next = new ListNode(0);
            pz = pz.next;
            diff--;
        }
        pz.next = len1 > len2 ? l2 : l1;

        ListNode lower = addTwoNumbersRecur(prime, zerosSentinel.next);
        if (this.carry == 1) {
            return new ListNode(1, lower);
        }
        return lower;
    }

    public ListNode addTwoNumbersRecur(ListNode l1, ListNode l2) {
        if (l1 == null) return null;
        ListNode lower = addTwoNumbersRecur(l1.next, l2.next);
        int addup = l1.val + l2.val + carry;
        ListNode curr = new ListNode(addup % 10, lower);
        this.carry = addup / 10;
        return curr;
    }

    
}
