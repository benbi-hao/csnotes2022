package leetcode;

import leetcode.ds.ListNode;
import leetcode.ds.TreeNode;
import java.lang.Math;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Queue;
import java.util.LinkedList;
import java.util.Stack;
import java.util.Collections;

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

    // 112. 路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        if (root.left == null && root.right == null) return targetSum == root.val;
        targetSum = targetSum - root.val;
        return hasPathSum(root.left, targetSum) || hasPathSum(root.right, targetSum);
    }

    // 437. 路径总和3
    private int numPathSum = 0;
    public int pathSum(TreeNode root, int targetSum) {
        pathSumRecur(root, targetSum);
        return numPathSum;
    }

    public void pathSumRecur(TreeNode root, int targetSum) {
        if (root == null) return;
        pathSumAsRoot(root, targetSum);
        pathSumRecur(root.left, targetSum);
        pathSumRecur(root.right, targetSum);
    }

    public void pathSumAsRoot(TreeNode root, long targetSum) {
        if (root == null) return;
        if (root.val == targetSum) { numPathSum += 1; }
        targetSum = targetSum - root.val;
        pathSumAsRoot(root.left, targetSum);
        pathSumAsRoot(root.right, targetSum);
    }

    // 572. 另一棵树的子树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) return false;
        return isSameTree(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    public boolean isSameTree(TreeNode root, TreeNode subRoot) {
        if (root == null) return subRoot == null;
        if (subRoot == null) return false;
        return root.val == subRoot.val && isSameTree(root.left, subRoot.left) && isSameTree(root.right, subRoot.right);
    }

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        return isSymmetric(root.left, root.right);
    }

    public boolean isSymmetric(TreeNode root1, TreeNode root2) {
        if (root1 == null) return root2 == null;
        if (root2 == null) return false;
        return root1.val == root2.val && isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }

    // 111. 二叉树的最小深度
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if (left == 0 || right == 0) return left + right + 1;
        return Math.min(left, right) + 1;
    }

    // 404. 左叶子之和
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
        int right = sumOfLeftLeaves(root.right);
        if (root.left == null) return right;
        int left = 0;
        if (root.left.left == null && root.left.right == null) {
            left = root.left.val;
        } else {
            left = sumOfLeftLeaves(root.left);
        }
        return right + left;
    }

    // 687. 最长同值路径
    // public int longestUnivaluePath(TreeNode root) {
    //     if (root == null) return 0;
    //     return Math.max(longestUnivaluePathAsRootPassed(root),
    //     Math.max(longestUnivaluePath(root.left), longestUnivaluePath(root.right)));
    // }

    // public int longestUnivaluePathAsRootPassed(TreeNode root) {
    //     if (root == null) return 0;
    //     return longestUnivaluePathAsRootValued(root.left, root.val) + longestUnivaluePathAsRootValued(root.right, root.val);
    // }

    // public int longestUnivaluePathAsRootValued(TreeNode root, int val) {
    //     if (root == null) return 0;
    //     if (root.val != val) return 0;
    //     return Math.max(longestUnivaluePathAsRootValued(root.left, val), longestUnivaluePathAsRootValued(root.right, val)) + 1;
    // }
    private int lengthLongestUnivaluePath;
    public int longestUnivaluePath(TreeNode root) {
        lengthLongestUnivaluePath = 0;
        longestUnivaluePathRecur(root);
        return lengthLongestUnivaluePath;
    }

    public int longestUnivaluePathRecur(TreeNode root) {
        if (root == null) return 0;
        int left = longestUnivaluePathRecur(root.left);
        int right = longestUnivaluePathRecur(root.right);
        left = (left != 0 && root.left.val != root.val) ? 0 : left;
        right = (right != 0 && root.right.val != root.val) ? 0 : right;
        lengthLongestUnivaluePath = Math.max(left + right, lengthLongestUnivaluePath);
        return Math.max(left, right) + 1;
    }

    // 337. 打家劫舍3 层次遍历
    private Map<TreeNode, Integer> robMemo;

    public int rob(TreeNode root) {
        robMemo = new HashMap<>();
        return robRecur(root);
    }

    public int robRecur(TreeNode root) {
        if (root == null) return 0;
        if (robMemo.containsKey(root)) return robMemo.get(root);
        int left = robRecur(root.left);
        int right = robRecur(root.right);
        int leftChildren = 0, rightChildren = 0;
        if (root.left != null) { leftChildren = robRecur(root.left.left) + robRecur(root.left.right); }
        if (root.right != null) { rightChildren = robRecur(root.right.left) + robRecur(root.right.right); }
        int ret = Math.max(left + right, root.val + leftChildren + rightChildren);
        robMemo.put(root, ret);
        return ret;
    }

    // 671. 二叉树中第二小的节点
    public int findSecondMinimumValue(TreeNode root) {
        if (root == null || root.left == null) return -1;
        int left = root.val == root.left.val ? findSecondMinimumValue(root.left) : root.left.val;
        int right = root.val == root.right.val ? findSecondMinimumValue(root.right) : root.right.val;
        if (left == -1) return right;
        if (right == -1) return left;
        return Math.min(left, right);
    }

    // - 层次遍历
    // 637. 二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> avgs = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode t = queue.poll();
                if (t.left != null) queue.add(t.left);
                if (t.right != null) queue.add(t.right);
                sum += t.val;
            }
            avgs.add(sum / size);
        }
        return avgs;
    }

    // 513. 找树左下角的值
    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        int bottomLeft = 0;
        queue.add(root);
        while(!queue.isEmpty()) {
            int size = queue.size();
            TreeNode t = queue.poll();
            if (t.left != null) queue.add(t.left);
            if (t.right != null) queue.add(t.right);
            bottomLeft = t.val;
            for (int i = 1; i < size; i++) {
                t = queue.poll();
                if (t.left != null) queue.add(t.left);
                if (t.right != null) queue.add(t.right);  
            } 
        }
        return bottomLeft;
    }

    // - 前中后序遍历
    // 144. 二叉树的前序表示
    // 尝试用迭代方式实现
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> traversalList = new ArrayList<>();
        if (root == null) return traversalList;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()) {
            TreeNode t = stack.pop();
            traversalList.add(t.val);
            if(t.right != null) stack.push(t.right);
            if(t.left != null) stack.push(t.left);
        }
        return traversalList;
    }

    // 145. 二叉树的后序遍历
    // 尝试用迭代方式实现
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> traversalList = new ArrayList<>();
        if (root == null) return traversalList;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()) {
            TreeNode t = stack.pop();
            traversalList.add(t.val);
            if (t.left != null) stack.push(t.left);
            if (t.right != null) stack.push(t.right);
        }
        Collections.reverse(traversalList);
        return traversalList;
    }

    // 94. 二叉树的中序遍历
    // 尝试用迭代方式实现
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> traversalList = new ArrayList<>();
        if (root == null) return traversalList;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            if (curr == null) {
                curr = stack.pop();
                traversalList.add(curr.val);
                curr = curr.right;
            }else {
                stack.push(curr);
                curr = curr.left;
            }
        }
        return traversalList;
    }

    // - BST
    // 669. 修剪二叉搜索树
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) return null;
        TreeNode left = null, right = null;
        if (root.val < high) right = trimBST(root.right, low, high);
        if (root.val > low) left = trimBST(root.left, low, high);
        if (root.val > high) return left;
        if (root.val < low) return right;
        root.left = left;
        root.right = right;
        return root;
    }

    
}
