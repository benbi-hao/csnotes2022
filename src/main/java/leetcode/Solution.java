package leetcode;

import leetcode.ds.ListNode;
import leetcode.ds.TreeNode;
import java.lang.Math;

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

    // 234. 回文链表
    // 要求O(1)空间复杂度 
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode slow = head, fast = head.next;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode half = slow.next;
        if (fast.next == null) slow.next = null;
        ListNode tail = reverseList(half);

        while (head != null) {
            if (head.val != tail.val) return false;
            head = head.next;
            tail = tail.next;
        }

        return true;
    }

    // 725. 分隔链表
    public ListNode[] splitListToParts(ListNode head, int k) {
        int len = 0;
        for (ListNode p = head; p != null; p = p.next) { len++; }
        int partLen = len / k;
        int mod = len % k;
        ListNode[] list = new ListNode[k];

        for (int i = 0; i < k; i++) {
            list[i] = head;
            if (i < mod) {
                head = cutList(head, partLen + 1);
            }else{
                head = cutList(head, partLen);
            }
        }

        return list;
    }

    public ListNode cutList(ListNode head, int len) {
        if (len == 0) return null;
        ListNode p = head;
        while (len > 1) {
            p = p.next;
            len--;
        }
        head = p.next;
        p.next = null;
        return head;
    }

    // 328. 奇偶链表
    public ListNode oddEvenList(ListNode head) {
        ListNode oddSentinel = new ListNode(0);
        ListNode evenSentinel = new ListNode(0);
        ListNode op = oddSentinel, ep = evenSentinel;

        ListNode curr = head, next = null;
        while(curr != null && curr.next != null) {
            next = curr.next.next;
            ep.next = curr.next;
            ep = ep.next;
            op.next = curr;
            op = op.next;
            curr = next;
        }
        
        op.next = null;
        ep.next = null;
        if (curr != null) {
            op.next = curr;
            op = op.next;
        }

        op.next = evenSentinel.next;
        return oddSentinel.next;
    }



    /**
     * 树
     */
    
    // - 递归
    // 104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // 110. 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return isBalancedRecur(root) == -1 ? false : true;
    }

    public int isBalancedRecur(TreeNode root) {
        if (root == null) return 0;
        int leftDepth = isBalancedRecur(root.left);
        if (leftDepth < 0) return -1;
        int rightDepth = isBalancedRecur(root.right);
        if (rightDepth < 0) return -1;
        int diff = leftDepth - rightDepth;
        if (diff < -1 || diff > 1) return -1;
        if (diff > 0) return leftDepth + 1;
        else return rightDepth + 1;
    }

    // 543. 二叉树的直径
    private int maxDiameter = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        diameterOfTreeAcross(root);
        return maxDiameter;
    }

    public int diameterOfTreeAcross(TreeNode root) {
        if (root == null) return 0;
        int leftDepth = diameterOfTreeAcross(root.left);
        int rightDepth = diameterOfTreeAcross(root.right);
        this.maxDiameter = Math.max(leftDepth + rightDepth, this.maxDiameter);
        return Math.max(leftDepth, rightDepth) + 1;
    }

    // 226. 翻转树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
        TreeNode left = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(left);
        return root;
    }

    // 617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) return root2;
        if (root2 == null) return root1;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        root1.val += root2.val;
        return root1;
    }

    
}
