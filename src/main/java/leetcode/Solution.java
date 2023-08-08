package leetcode;

import leetcode.ds.ListNode;
import leetcode.ds.TreeNode;
import leetcode.util.UnionFind;

import java.lang.Math;
import java.util.*;

public class Solution {
    private Random random = new Random();

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
    public ListNode reverseList(ListNode head) { // 迭代法
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

    public ListNode reverseListRecur(ListNode head) { // 递归法
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverseListRecur(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    public ListNode reverseListInsert(ListNode head) { // 头插法
        ListNode sentinel = new ListNode(0);
        ListNode curr = head, next = null;
        while (curr != null) {
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
    public ListNode deleteDuplicates(ListNode head) { // 迭代法
        if (head == null)
            return null;
        ListNode curr = head;
        while (curr.next != null) {
            if (curr.val == curr.next.val) {
                curr.next = curr.next.next;
            } else {
                curr = curr.next;
            }
        }
        return head;
    }

    public ListNode deleteDuplicatesRecur(ListNode head) { // 递归法
        if (head == null || head.next == null) {
            return head;
        }
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
    public ListNode swapPairs(ListNode head) { // 递归法
        if (head == null || head.next == null)
            return head;
        head.next.next = swapPairs(head.next.next);
        ListNode ret = head.next;
        head.next = ret.next;
        ret.next = head;
        return ret;
    }

    public ListNode swapPairsIter(ListNode head) { // 迭代法
        if (head == null || head.next == null)
            return head;
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

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) { // 不反转链表
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
        while (diff > 0) {
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
        if (l1 == null)
            return null;
        ListNode lower = addTwoNumbersRecur(l1.next, l2.next);
        int addup = l1.val + l2.val + carry;
        ListNode curr = new ListNode(addup % 10, lower);
        this.carry = addup / 10;
        return curr;
    }

    // 234. 回文链表
    // 要求O(1)空间复杂度
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null)
            return true;
        ListNode slow = head, fast = head.next;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode half = slow.next;
        if (fast.next == null)
            slow.next = null;
        ListNode tail = reverseList(half);

        while (head != null) {
            if (head.val != tail.val)
                return false;
            head = head.next;
            tail = tail.next;
        }

        return true;
    }

    // 725. 分隔链表
    public ListNode[] splitListToParts(ListNode head, int k) {
        int len = 0;
        for (ListNode p = head; p != null; p = p.next) {
            len++;
        }
        int partLen = len / k;
        int mod = len % k;
        ListNode[] list = new ListNode[k];

        for (int i = 0; i < k; i++) {
            list[i] = head;
            if (i < mod) {
                head = cutList(head, partLen + 1);
            } else {
                head = cutList(head, partLen);
            }
        }

        return list;
    }

    public ListNode cutList(ListNode head, int len) {
        if (len == 0)
            return null;
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
        while (curr != null && curr.next != null) {
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
        if (root == null)
            return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // 110. 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return isBalancedRecur(root) == -1 ? false : true;
    }

    public int isBalancedRecur(TreeNode root) {
        if (root == null)
            return 0;
        int leftDepth = isBalancedRecur(root.left);
        if (leftDepth < 0)
            return -1;
        int rightDepth = isBalancedRecur(root.right);
        if (rightDepth < 0)
            return -1;
        int diff = leftDepth - rightDepth;
        if (diff < -1 || diff > 1)
            return -1;
        if (diff > 0)
            return leftDepth + 1;
        else
            return rightDepth + 1;
    }

    // 543. 二叉树的直径
    private int maxDiameter = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        diameterOfTreeAcross(root);
        return maxDiameter;
    }

    public int diameterOfTreeAcross(TreeNode root) {
        if (root == null)
            return 0;
        int leftDepth = diameterOfTreeAcross(root.left);
        int rightDepth = diameterOfTreeAcross(root.right);
        this.maxDiameter = Math.max(leftDepth + rightDepth, this.maxDiameter);
        return Math.max(leftDepth, rightDepth) + 1;
    }

    // 226. 翻转树
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return root;
        TreeNode left = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(left);
        return root;
    }

    // 617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null)
            return root2;
        if (root2 == null)
            return root1;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        root1.val += root2.val;
        return root1;
    }

    // 112. 路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null)
            return false;
        if (root.left == null && root.right == null)
            return targetSum == root.val;
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
        if (root == null)
            return;
        pathSumAsRoot(root, targetSum);
        pathSumRecur(root.left, targetSum);
        pathSumRecur(root.right, targetSum);
    }

    public void pathSumAsRoot(TreeNode root, long targetSum) {
        if (root == null)
            return;
        if (root.val == targetSum) {
            numPathSum += 1;
        }
        targetSum = targetSum - root.val;
        pathSumAsRoot(root.left, targetSum);
        pathSumAsRoot(root.right, targetSum);
    }

    // 572. 另一棵树的子树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null)
            return false;
        return isSameTree(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    public boolean isSameTree(TreeNode root, TreeNode subRoot) {
        if (root == null)
            return subRoot == null;
        if (subRoot == null)
            return false;
        return root.val == subRoot.val && isSameTree(root.left, subRoot.left) && isSameTree(root.right, subRoot.right);
    }

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        return isSymmetric(root.left, root.right);
    }

    public boolean isSymmetric(TreeNode root1, TreeNode root2) {
        if (root1 == null)
            return root2 == null;
        if (root2 == null)
            return false;
        return root1.val == root2.val && isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }

    // 111. 二叉树的最小深度
    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if (left == 0 || right == 0)
            return left + right + 1;
        return Math.min(left, right) + 1;
    }

    // 404. 左叶子之和
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null)
            return 0;
        int right = sumOfLeftLeaves(root.right);
        if (root.left == null)
            return right;
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
    // if (root == null) return 0;
    // return Math.max(longestUnivaluePathAsRootPassed(root),
    // Math.max(longestUnivaluePath(root.left), longestUnivaluePath(root.right)));
    // }

    // public int longestUnivaluePathAsRootPassed(TreeNode root) {
    // if (root == null) return 0;
    // return longestUnivaluePathAsRootValued(root.left, root.val) +
    // longestUnivaluePathAsRootValued(root.right, root.val);
    // }

    // public int longestUnivaluePathAsRootValued(TreeNode root, int val) {
    // if (root == null) return 0;
    // if (root.val != val) return 0;
    // return Math.max(longestUnivaluePathAsRootValued(root.left, val),
    // longestUnivaluePathAsRootValued(root.right, val)) + 1;
    // }
    private int lengthLongestUnivaluePath;

    public int longestUnivaluePath(TreeNode root) {
        lengthLongestUnivaluePath = 0;
        longestUnivaluePathRecur(root);
        return lengthLongestUnivaluePath;
    }

    public int longestUnivaluePathRecur(TreeNode root) {
        if (root == null)
            return 0;
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
        if (root == null)
            return 0;
        if (robMemo.containsKey(root))
            return robMemo.get(root);
        int left = robRecur(root.left);
        int right = robRecur(root.right);
        int leftChildren = 0, rightChildren = 0;
        if (root.left != null) {
            leftChildren = robRecur(root.left.left) + robRecur(root.left.right);
        }
        if (root.right != null) {
            rightChildren = robRecur(root.right.left) + robRecur(root.right.right);
        }
        int ret = Math.max(left + right, root.val + leftChildren + rightChildren);
        robMemo.put(root, ret);
        return ret;
    }

    // 671. 二叉树中第二小的节点
    public int findSecondMinimumValue(TreeNode root) {
        if (root == null || root.left == null)
            return -1;
        int left = root.val == root.left.val ? findSecondMinimumValue(root.left) : root.left.val;
        int right = root.val == root.right.val ? findSecondMinimumValue(root.right) : root.right.val;
        if (left == -1)
            return right;
        if (right == -1)
            return left;
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
                if (t.left != null)
                    queue.add(t.left);
                if (t.right != null)
                    queue.add(t.right);
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
        while (!queue.isEmpty()) {
            int size = queue.size();
            TreeNode t = queue.poll();
            if (t.left != null)
                queue.add(t.left);
            if (t.right != null)
                queue.add(t.right);
            bottomLeft = t.val;
            for (int i = 1; i < size; i++) {
                t = queue.poll();
                if (t.left != null)
                    queue.add(t.left);
                if (t.right != null)
                    queue.add(t.right);
            }
        }
        return bottomLeft;
    }

    // - 前中后序遍历
    // 144. 二叉树的前序表示
    // 尝试用迭代方式实现
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> traversalList = new ArrayList<>();
        if (root == null)
            return traversalList;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode t = stack.pop();
            traversalList.add(t.val);
            if (t.right != null)
                stack.push(t.right);
            if (t.left != null)
                stack.push(t.left);
        }
        return traversalList;
    }

    // 145. 二叉树的后序遍历
    // 尝试用迭代方式实现
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> traversalList = new ArrayList<>();
        if (root == null)
            return traversalList;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode t = stack.pop();
            traversalList.add(t.val);
            if (t.left != null)
                stack.push(t.left);
            if (t.right != null)
                stack.push(t.right);
        }
        Collections.reverse(traversalList);
        return traversalList;
    }

    // 94. 二叉树的中序遍历
    // 尝试用迭代方式实现
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> traversalList = new ArrayList<>();
        if (root == null)
            return traversalList;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            if (curr == null) {
                curr = stack.pop();
                traversalList.add(curr.val);
                curr = curr.right;
            } else {
                stack.push(curr);
                curr = curr.left;
            }
        }
        return traversalList;
    }

    // - BST
    // 669. 修剪二叉搜索树
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null)
            return null;
        TreeNode left = null, right = null;
        if (root.val < high)
            right = trimBST(root.right, low, high);
        if (root.val > low)
            left = trimBST(root.left, low, high);
        if (root.val > high)
            return left;
        if (root.val < low)
            return right;
        root.left = left;
        root.right = right;
        return root;
    }

    // 230. 二叉搜索树中第k小的元素
    private int kthSmallestCnt;
    private int kthSmallestRet;

    public int kthSmallest(TreeNode root, int k) { // 中序遍历，常数空间复杂度
        kthSmallestCnt = 0;
        kthSmallestRecur(root, k);
        return kthSmallestRet;
    }

    public void kthSmallestRecur(TreeNode root, int k) {
        if (root == null)
            return;
        kthSmallestRecur(root.left, k);
        if (kthSmallestCnt >= k) {
            return;
        } else {
            kthSmallestRet = root.val;
            kthSmallestCnt += 1;
        }
        kthSmallestRecur(root.right, k);
    }

    // 538. 把二叉搜索树转换为累加树
    private int convertBSTSum;

    public TreeNode convertBST(TreeNode root) { // 反向中序遍历
        convertBSTSum = 0;
        convertBSTRecur(root);
        return root;
    }

    public void convertBSTRecur(TreeNode root) {
        if (root == null)
            return;
        convertBSTRecur(root.right);
        root.val += convertBSTSum;
        convertBSTSum = root.val;
        convertBSTRecur(root.left);
    }

    // 235. 二叉搜索树的最近公共祖先
    public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null)
            return root;
        int hi, lo;
        if (p.val > q.val) {
            hi = p.val;
            lo = q.val;
        } else {
            hi = q.val;
            lo = p.val;
        }
        return lowestCommonAncestorBST(root, lo, hi);
    }

    public TreeNode lowestCommonAncestorBST(TreeNode root, int lo, int hi) {
        if (root == null)
            return root;
        if (root.val >= lo && root.val <= hi)
            return root;
        else if (root.val > hi)
            return lowestCommonAncestorBST(root.left, lo, hi);
        else
            return lowestCommonAncestorBST(root.right, lo, hi);
    }

    // 236. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestorT(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root.val == p.val || root.val == q.val)
            return root;
        TreeNode left = lowestCommonAncestorT(root.left, p, q);
        TreeNode right = lowestCommonAncestorT(root.right, p, q);
        if (left == null)
            return right;
        if (right == null)
            return left;
        return root;
    }

    // 108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayToBST(int[] nums, int lo, int hi) {
        if (lo > hi)
            return null;
        int mid = (lo + hi) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, lo, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, hi);
        return root;
    }

    // 109. 有序链表转换二叉搜索树
    public TreeNode sortedListToBST(ListNode head) { // 空间换时间
        int len = 0;
        for (ListNode p = head; p != null; p = p.next)
            len++;
        int[] nums = new int[len];
        ListNode p = head;
        for (int i = 0; i < len; i++) {
            nums[i] = p.val;
            p = p.next;
        }
        return sortedArrayToBST(nums, 0, len - 1);
    }

    // 653. 两数之和IV-输入BST
    private Set<Integer> findTargetSet;
    private boolean findTargetFlag;
    private int findTargetSum;

    public boolean findTarget(TreeNode root, int k) { // 使用集合
        findTargetSet = new HashSet<>();
        findTargetFlag = false;
        findTargetSum = k;
        return findTargetTravesal(root);
    }

    public boolean findTargetTravesal(TreeNode root) {
        if (root == null)
            return false;
        if (findTargetFlag)
            return true;
        if (findTargetSet.contains(root.val)) {
            findTargetFlag = true;
            return true;
        }
        findTargetSet.add(findTargetSum - root.val);
        return findTargetTravesal(root.left) || findTargetTravesal(root.right);
    }

    private List<Integer> findTargetList;

    public boolean findTargetArray(TreeNode root, int k) { // 利用BST中序遍历有序特性（更快）
        findTargetList = new ArrayList<>();
        findTargetInorder(root);
        int lo = 0, hi = findTargetList.size() - 1;
        while (lo < hi) {
            int sum = findTargetList.get(lo) + findTargetList.get(hi);
            if (sum == k)
                return true;
            else if (sum < k)
                lo++;
            else
                hi--;
        }
        return false;
    }

    public void findTargetInorder(TreeNode root) {
        if (root == null)
            return;
        findTargetInorder(root.left);
        findTargetList.add(root.val);
        findTargetInorder(root.right);
    }

    // 530. 二叉搜索树的最小绝对差
    private int getMinimumDifferenceCurr;
    private int getMinimumDifferenceMinDiff;

    public int getMinimumDifference(TreeNode root) { // O(1)空间复杂度，比读成数组好
        getMinimumDifferenceCurr = -1;
        getMinimumDifferenceMinDiff = Integer.MAX_VALUE;
        getMinimumDifferenceInorder(root);
        return getMinimumDifferenceMinDiff;
    }

    public void getMinimumDifferenceInorder(TreeNode root) {
        if (root == null)
            return;
        getMinimumDifferenceInorder(root.left);
        if (getMinimumDifferenceCurr != -1) {
            getMinimumDifferenceMinDiff = Math.min(getMinimumDifferenceMinDiff, root.val - getMinimumDifferenceCurr);
        }
        getMinimumDifferenceCurr = root.val;
        getMinimumDifferenceInorder(root.right);
    }

    // 501. 二叉搜索树中的众数
    private int findModeCurr;
    private int findModeCurrCnt;
    private int findModeMostCnt;
    private List<Integer> findModeList;

    public int[] findMode(TreeNode root) {
        findModeCurr = Integer.MIN_VALUE;
        findModeCurrCnt = 0;
        findModeMostCnt = 0;
        findModeList = new ArrayList<>();
        findModeInorder(root);
        // 收尾
        if (findModeCurrCnt >= findModeMostCnt) {
            if (findModeCurrCnt > findModeMostCnt) {
                findModeMostCnt = findModeCurrCnt;
                findModeList.clear();
            }
            findModeList.add(findModeCurr);
        }
        int[] ret = new int[findModeList.size()];
        int i = 0;
        for (int num : findModeList) {
            ret[i++] = num;
        }
        return ret;
        // 用stream转换数组虽然只需要一行，但是速度非常慢
        // return findModeList.stream().mapToInt(Integer::valueOf).toArray();
    }

    public void findModeInorder(TreeNode root) {
        if (root == null)
            return;
        findModeInorder(root.left);
        if (findModeCurr == root.val) {
            findModeCurrCnt++;
        } else {
            if (findModeCurrCnt >= findModeMostCnt) {
                if (findModeCurrCnt > findModeMostCnt) {
                    findModeMostCnt = findModeCurrCnt;
                    findModeList.clear();
                }
                findModeList.add(findModeCurr);
            }
            findModeCurr = root.val;
            findModeCurrCnt = 1;
        }
        findModeInorder(root.right);
    }

    // - Trie
    // 208. 实现Trie（前缀树）
    // 见leetcode.ds.Trie

    // 677. 键值映射
    // 见leetcode.ds.MapSum

    /**
     * 栈和队列
     */

    // 232. 用栈实现队列
    // 见leetcode.ds.MyQueue

    // 225. 用队列实现栈
    // 见leetcode.ds.MyStack

    // 155. 最小栈
    // 见leetcode.ds.MinStack

    // 20. 有效的括号
    public boolean isValid(String s) {
        int len = s.length();
        if (len % 2 == 1)
            return false;
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else {
                if (stack.isEmpty() || stack.pop() != isValidLeftOf(c))
                    return false;
            }
        }
        return stack.isEmpty();
    }

    private char isValidLeftOf(char r) {
        switch (r) {
            case ')':
                return '(';
            case ']':
                return '[';
            case '}':
                return '{';
            default:
                return ' ';
        }
    }

    // 739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) { // 反向遍历，栈
        int len = temperatures.length;
        int[] ret = new int[len];
        Stack<Integer> stack = new Stack<>();
        for (int i = len - 1; i >= 0; i--) {
            while (!stack.isEmpty() && temperatures[stack.peek()] <= temperatures[i])
                stack.pop();
            if (stack.isEmpty())
                ret[i] = 0;
            else
                ret[i] = stack.peek() - i;
            stack.push(i);
        }
        return ret;
    }

    public int[] dailyTemperaturesForward(int[] temperatures) { // 正向遍历，栈（优于反向，因为遵循了栈内存放待解决问题的思路）
        int len = temperatures.length;
        int[] ret = new int[len];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                int prev = stack.pop();
                ret[prev] = i - prev;
            }
            stack.push(i);
        }
        return ret;
    }

    public int[] dailyTemperaturesBrute(int[] temperatures) {
        int n = temperatures.length;
        int[] ret = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (temperatures[i] < temperatures[j]) {
                    ret[i] = j - i;
                    break;
                }
            }
        }
        return ret;
    }

    // 503. 下一个更大的元素2
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ret = new int[n];
        Stack<Integer> stack = new Stack<>();
        int max = nums[0];
        int maxIndex = 0;
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                int prev = stack.pop();
                ret[prev] = nums[i];
            }
            stack.push(i);
            if (nums[i] > max) {
                max = nums[i];
                maxIndex = i;
            }
        }
        for (int i = 0; i <= maxIndex; i++) {
            while (nums[stack.peek()] < nums[i]) {
                int prev = stack.pop();
                ret[prev] = nums[i];
            }
        }
        while (!stack.isEmpty()) {
            ret[stack.pop()] = -1;
        }
        return ret;
    }

    /**
     * 哈希表
     */
    // 1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int corr = target - nums[i];
            if (map.containsKey(corr))
                return new int[] { map.get(corr), i };
            map.put(nums[i], i);
        }
        return null;
    }

    // 217. 存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num))
                return true;
            set.add(num);
        }
        return false;
    }

    // 594. 最长和谐子序列
    public int findLHS(int[] nums) {
        Map<Integer, Integer> freqMap = new HashMap<>();
        for (int num : nums) {
            freqMap.put(num, freqMap.getOrDefault(num, 0) + 1);
        }
        int ret = 0;
        // 增强for循环和ForEach和iterator本质上是一个东西
        for (int key : freqMap.keySet()) {
            if (freqMap.containsKey(key + 1)) {
                ret = Math.max(ret, freqMap.get(key) + freqMap.get(key + 1));
            }
        }
        return ret;
    }

    // 128. 最长连续序列
    private Map<Integer, Integer> longestConsecutiveLengthMap;
    private int longestConsecutiveMax;

    public int longestConsecutive(int[] nums) {
        longestConsecutiveMax = 0;
        longestConsecutiveLengthMap = new HashMap<>();
        for (int num : nums) {
            longestConsecutiveLengthMap.put(num, 0);
        }
        for (int num : nums) {
            longestConsecutiveMax = Math.max(longestConsecutiveMax, longestConsecutiveRecur(num));
        }
        return longestConsecutiveMax;
    }

    public int longestConsecutiveRecur(int head) {
        if (!longestConsecutiveLengthMap.containsKey(head)) {
            return 0;
        }
        int length = longestConsecutiveLengthMap.get(head);
        if (length != 0) {
            return length;
        }
        length = longestConsecutiveRecur(head + 1) + 1;
        longestConsecutiveLengthMap.put(head, length);
        return length;
    }

    /**
     * 字符串
     */
    // 字符串循环移位包含
    public boolean circleSubString(String s1, String s2) {
        return (s1 + s1).contains(s2);
    }

    // 字符串循环移位
    public String circleShift(String s, int k) {
        char[] cs = s.toCharArray();
        reverseCharsequence(cs, 0, k - 1);
        reverseCharsequence(cs, k, cs.length - 1);
        reverseCharsequence(cs, 0, cs.length - 1);
        return new String(cs);
    }

    private void reverseCharsequence(char[] cs, int lo, int hi) {
        while (lo < hi) {
            char t = cs[lo];
            cs[lo++] = cs[hi];
            cs[hi--] = t;
        }
    }

    // 字符串中单词的翻转
    public String reverseWords(String s) {
        String[] words = s.split(" ");
        char[] cs = s.toCharArray();
        int index = 0;
        for (String word : words) {
            reverseCharsequence(cs, index, word.length() - 1);
            index += word.length() + 1;
        }
        reverseCharsequence(cs, 0, cs.length);
        return new String(cs);
    }

    // 242. 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length())
            return false;
        int n = s.length();
        int[] freqs = new int[26];
        for (int i = 0; i < n; i++) {
            freqs[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < n; i++) {
            freqs[t.charAt(i) - 'a']--;
        }
        for (int freq : freqs) {
            if (freq != 0)
                return false;
        }
        return true;
    }

    // 409. 最长回文串
    public int longestPalindromeIter(String s) {
        int[] upperCaseFreqs = new int[26];
        int[] lowerCaseFreqs = new int[26];
        int n = s.length();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c >= 'A' && c <= 'Z') {
                upperCaseFreqs[c - 'A']++;
            } else {
                lowerCaseFreqs[c - 'a']++;
            }
        }
        int maxLength = 0;
        for (int freq : upperCaseFreqs) {
            maxLength += freq;
            if (freq % 2 != 0) {
                maxLength -= 1;
            }
        }
        for (int freq : lowerCaseFreqs) {
            maxLength += freq;
            if (freq % 2 != 0) {
                maxLength -= 1;
            }
        }
        return maxLength == n ? maxLength : maxLength + 1;
    }

    // 205. 同构字符串
    public boolean isIsomorphic(String s, String t) {
        char[] smap = new char[256];
        char[] tmap = new char[256];
        int n = s.length();
        for (int i = 0; i < n; i++) {
            char sc = s.charAt(i);
            char tc = t.charAt(i);
            if (smap[sc] == '\0' && tmap[tc] == '\0') {
                smap[sc] = tc;
                tmap[tc] = sc;
            } else {
                if (smap[sc] != tc)
                    return false;
            }
        }
        return true;
    }

    // 647. 回文子串
    private int countSubstringsCnt;

    public int countSubstringsIter(String s) { // 迭代且不使用额外空间
        int n = s.length();
        for (int i = 0; i < n; i++) {
            countSubstringsExtend(s, i, i);
            countSubstringsExtend(s, i, i + 1);
        }
        return countSubstringsCnt;
    }

    private void countSubstringsExtend(String s, int l, int r) {
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            countSubstringsCnt++;
            l--;
            r++;
        }
    }

    private byte[][] countSubstringsResults;

    public int countSubstringsRecur(String s) { // 动态规划
        int n = s.length();
        countSubstringsResults = new byte[n][n];
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (countSubstringsIsPal(s, i, j)) {
                    cnt++;
                }
            }
        }
        return cnt;
    }

    private boolean countSubstringsIsPal(String s, int l, int r) {
        if (l > r)
            return true;
        if (countSubstringsResults[l][r] != 0)
            return countSubstringsResults[l][r] == 1;
        boolean ret = s.charAt(l) == s.charAt(r) && countSubstringsIsPal(s, l + 1, r - 1);
        countSubstringsResults[l][r] = (byte) (ret ? 1 : 2);
        return ret;
    }

    // 9. 回文数
    public boolean isPalindromeIter(int x) { // 算位数然后迭代
        if (x < 0)
            return false;
        int t = x;
        int n = 0;
        while (t != 0) {
            n++;
            t /= 10;
        }
        int lo, hi;
        lo = n / 2 - 1;
        if (n % 2 == 0) {
            hi = n / 2;
        } else {
            hi = n / 2 + 1;
        }
        while (lo >= 0) {
            if (isPalindromeDigitOf(x, lo--) != isPalindromeDigitOf(x, hi++)) {
                return false;
            }
        }
        return true;
    }

    private int isPalindromeDigitOf(int num, int digit) {
        return (num / (int) Math.pow(10, digit)) % 10;
    }

    public boolean isPalindrome(int x) { // 利用整数运算，比较int高位和低位翻转
        if (x == 0)
            return true;
        if (x < 0 || x % 10 == 0)
            return false;
        int right = 0;
        while (x > right) {
            right = right * 10 + x % 10;
            x /= 10;
        }
        return x == right || x == right / 10;
    }

    // 696. 计数二进制字符串
    public int countBinarySubstrings(String s) {
        int cnt = 0;
        int n = s.length();
        char curr = s.charAt(0);
        int currSecLen = 1;
        int prevSecLen = 0;
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) != curr) {
                cnt += Math.min(prevSecLen, currSecLen);
                curr = s.charAt(i);
                prevSecLen = currSecLen;
                currSecLen = 1;
            } else {
                currSecLen++;
            }
        }
        cnt += Math.min(prevSecLen, currSecLen);
        return cnt;
    }

    /**
     * 数组与矩阵
     */
    // 283. 移动零
    public void moveZeros(int[] nums) {
        int n = nums.length;
        int i = 0;
        for (int num : nums) {
            if (num != 0) {
                nums[i++] = num;
            }
        }
        while (i < n) {
            nums[i++] = 0;
        }
    }

    // 566. 重塑矩阵
    public int[][] matrixReshape(int[][] mat, int r, int c) {
        int m = mat.length, n = mat[0].length;
        int dim = m * n;
        if (r * c != dim)
            return mat;
        int[][] ret = new int[r][c];
        int i = 0, j = 0;
        for (int[] row : mat) {
            for (int num : row) {
                ret[i][j] = num;
                if (j == c - 1) {
                    i += 1;
                    j = 0;
                } else {
                    j += 1;
                }
            }
        }
        return ret;
    }

    // 485. 最大连续1的个数
    public int findMaxConsecutiveOnes(int[] nums) {
        int ret = 0;
        int currLen = 0;
        for (int num : nums) {
            currLen = num == 1 ? currLen + 1 : 0;
            ret = Math.max(ret, currLen);
        }
        return ret;
    }

    // 240. 搜索二维矩阵2
    public boolean searchMatrix(int[][] matrix, int target) { // 过程相当于一步步将目标空间缩小
        int m = matrix.length, n = matrix[0].length;
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target)
                return true;
            else if (matrix[i][j] < target)
                i++;
            else
                j--;
        }
        return false;
    }

    // 378. 有序矩阵中第K小的元素
    public int kthSmallest(int[][] matrix, int k) {
        int m = matrix.length, n = matrix[0].length;
        int lo = matrix[0][0], hi = matrix[m - 1][n - 1];
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int cnt = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n && matrix[i][j] <= mid; j++) {
                    cnt++;
                }
            }
            if (cnt < k)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    // 645. 错误的集合
    public int[] findErrorNums(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] != i + 1 && nums[i] != nums[nums[i] - 1]) {
                findErrorNumsSwap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1)
                return new int[] { nums[i], i + 1 };
        }
        return null;
    }

    private void findErrorNumsSwap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    // 287. 寻找重复数
    public int findDuplicateBin(int[] nums) { // 在取值空间上做二分查找，相当于找间隔右侧
        int n = nums.length;
        int l = 1, r = n - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) >> 1;
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                if (nums[i] <= mid) {
                    cnt++;
                }
            }
            if (cnt <= mid) {
                l = mid + 1;
            } else {
                r = mid - 1;
                ans = mid;
            }
        }
        return ans;
    }

    public int findDuplicate(int[] nums) { // 问题转换
        int slow = nums[0], fast = nums[nums[0]];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    // 667. 优美的排列
    public int[] constructArray(int n, int k) {
        int[] array = new int[n];
        int idx = 0;
        int lo = 1, hi = n;
        while (k > 0) {
            if (idx % 2 == 0) {
                array[idx] = lo++;
            } else {
                array[idx] = hi--;
            }
            k--;
            idx++;
        }
        if (array[idx - 1] < hi) {
            for (int i = lo; i <= hi; i++) {
                array[idx++] = i;
            }
        } else {
            for (int i = hi; i >= lo; i--) {
                array[idx++] = i;
            }
        }
        return array;
    }

    // 697. 数组的度
    public int findShortestSubArray(int[] nums) {
        int n = nums.length;
        int maxFreq = 0;
        int retLen = n + 1;
        Map<Integer, Integer> freqs = new HashMap<>();
        Map<Integer, Integer> leftPos = new HashMap<>();
        Map<Integer, Integer> rightPos = new HashMap<>();
        for (int i = 0; i < n; i++) {
            freqs.put(nums[i], freqs.getOrDefault(nums[i], 0) + 1);
            if (!leftPos.containsKey(nums[i]))
                leftPos.put(nums[i], i);
            rightPos.put(nums[i], i);
        }
        Set<Map.Entry<Integer, Integer>> entrySet = freqs.entrySet();
        for (Map.Entry<Integer, Integer> entry : entrySet) {
            maxFreq = Math.max(maxFreq, entry.getValue());
        }
        for (Map.Entry<Integer, Integer> entry : entrySet) {
            int freq = entry.getValue();
            if (freq == maxFreq) {
                int key = entry.getKey();
                int len = rightPos.get(key) - leftPos.get(key) + 1;
                if (len < retLen) {
                    retLen = len;
                }
            }
        }
        return retLen;
    }

    // 766. 托普利茨矩阵
    public boolean isToeplitzMatrix(int[][] matrix) {
        for (int i = 0; i < matrix.length - 1; i++) {
            if (!isToeplitzMatrixCompare(matrix[i], matrix[i + 1]))
                return false;
        }
        return true;
    }

    private boolean isToeplitzMatrixCompare(int[] overLine, int[] underLine) {
        for (int i = 0; i < overLine.length - 1; i++) {
            if (overLine[i] != underLine[i + 1])
                return false;
        }
        return true;
    }

    // 565. 数组嵌套
    public int arrayNesting(int[] nums) {
        int maxSize = 1;
        for (int i = 0; i < nums.length; i++) {
            int size = 0;
            int j = i;
            while (nums[j] != -1) {
                int t = j;
                j = nums[j];
                nums[t] = -1;
                size++;
            }
            maxSize = Math.max(maxSize, size);
        }
        return maxSize;
    }

    // 769. 最多能完成排序的块
    public int maxChunksToSorted(int[] arr) {
        int chunkCnt = 0;
        int max = arr[0];
        for (int i = 0; i < arr.length; i++) {
            max = Math.max(max, arr[i]);
            if (max == i) {
                chunkCnt++;
            }
        }
        return chunkCnt;
    }

    /**
     * 图
     */
    // - 二分图
    // 785. 判断二分图
    private int[] isBipartiteColors;

    public boolean isBipartite(int[][] graph) {
        isBipartiteColors = new int[graph.length];
        for (int i = 0; i < graph.length; i++) {
            if (isBipartiteColors[i] == 0 && !isBipartite(graph, i, 1)) {
                return false;
            }
        }
        return true;
    }

    private boolean isBipartite(int[][] graph, int v, int color) { // 将上色和判断顺便放在一起，最开始的思路是将上色和判断分开来，但是比这个方法慢
        if (isBipartiteColors[v] != 0) {
            return isBipartiteColors[v] == color;
        }
        isBipartiteColors[v] = color;
        for (int u : graph[v]) {
            if (!isBipartite(graph, u, -color)) {
                return false;
            }
        }
        return true;
    }

    // - 拓扑排序
    // 207. 课程表
    private boolean[] canFinishGlobalMark;
    private boolean[] canFinishLocalMark;

    public boolean canFinish(int numCourses, int[][] prerequisites) { // 记录dfs递归路径，如果路径折返了说明有环
        canFinishGlobalMark = new boolean[numCourses];
        canFinishLocalMark = new boolean[numCourses];
        List<Integer>[] graph = new List[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] req : prerequisites) {
            graph[req[1]].add(req[0]);
        }
        for (int i = 0; i < numCourses; i++) {
            if (canFinishHasCycle(graph, i))
                return false;
        }
        return true;
    }

    public boolean canFinishHasCycle(List<Integer>[] graph, int curr) {
        if (canFinishLocalMark[curr])
            return true;
        if (canFinishGlobalMark[curr])
            return false;
        canFinishGlobalMark[curr] = true;
        canFinishLocalMark[curr] = true;
        for (int next : graph[curr]) {
            if (canFinishHasCycle(graph, next))
                return true;
        }
        canFinishLocalMark[curr] = false;
        return false;
    }

    // 210. 课程表2
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        List<Integer>[] sparseGraph = new List[numCourses];
        int[] order = new int[numCourses];
        int idx = 0;
        for (int i = 0; i < numCourses; i++) {
            sparseGraph[i] = new ArrayList<>();
        }
        for (int[] req : prerequisites) {
            indegrees[req[0]]++;
            sparseGraph[req[1]].add(req[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegrees[i] == 0)
                queue.offer(i);
        }
        while (!queue.isEmpty()) {
            int curr = queue.poll();
            for (int next : sparseGraph[curr]) {
                if (--indegrees[next] == 0) {
                    queue.offer(next);
                }
            }
            order[idx++] = curr;
        }
        return idx == numCourses ? order : new int[0];
    }

    // - 并查集
    // 684. 冗余连接
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        UnionFind uf = new UnionFind(n);
        for (int[] edge : edges) {
            if (uf.isConnected(edge[0], edge[1]))
                return edge;
            uf.union(edge[0], edge[1]);
        }
        return null;
    }

    /**
     * 位运算
     */
    // x ^ 0s = x x & 0s = 0 x | 0s = x
    // x ^ 1s = ~x x & 1s = x x | 1s = 1s
    // x ^ x = 0 x & x = x x | x = x

    // 461. 汉明距离
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }

    // 136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int ret = 0;
        for (int num : nums) {
            ret ^= num;
        }
        return ret;
    }

    // 268. 丢失的数字
    public int missingNumber(int[] nums) {
        int ret = 0;
        for (int i = 0; i < nums.length; i++) {
            ret = ret ^ i ^ nums[i];
        }
        return ret ^ nums.length;
    }

    // 260. 只出现一次数字3
    public int[] singleNumber3(int[] nums) {
        int[] ret = new int[2];
        int diff = 0;
        for (int num : nums) {
            diff ^= num;
        }
        diff &= -diff;
        for (int num : nums) {
            if ((num & diff) == 0)
                ret[0] ^= num;
            else
                ret[1] ^= num;
        }
        return ret;
    }

    // 190. 颠倒二进制位
    public int reverseBits(int n) {
        int ret = 0;
        for (int i = 0; i < 32; i++) {
            ret <<= 1;
            ret |= (n & 1);
            n >>>= 1;
        }
        return ret;
    }

    // 不用额外变量交换两个数
    public void swap(int a, int b) {
        a = a ^ b;
        b = a ^ b;
        a = a ^ b;
    }

    // 231. 2的幂
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // 342. 4的幂
    public boolean isPowerOfFour(int num) {
        return num > 0 && (num & (num - 1)) == 0 && (num & 0x55555555) != 0;
    }

    // 693. 交替位二进制数
    public boolean hasAlternatingBits(int n) {
        int a = (n ^ (n >> 1));
        return (a & (a + 1)) == 0;
    }

    // 476. 数字的补数
    public int findComplement(int num) {
        if (num == 0)
            return 1;
        int mask = 1 << 30;
        while ((num & mask) == 0)
            mask >>= 1;
        mask = (mask << 1) - 1;
        return num ^ mask;
    }

    public int findComplementB(int num) {
        int mask = num;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        return (mask ^ num);
    }

    // 371. 两整数之和
    public int getSum(int a, int b) {
        return b == 0 ? a : getSum((a ^ b), (a & b) << 1);
    }

    // 318. 最大单词长度乘积
    public int maxProduct(String[] words) {
        int n = words.length;
        int[] val = new int[n];
        for (int i = 0; i < n; i++) {
            for (char c : words[i].toCharArray()) {
                val[i] |= 1 << (c - 'a');
            }
        }
        int ret = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((val[i] & val[j]) == 0) {
                    ret = Math.max(ret, words[i].length() * words[j].length());
                }
            }
        }
        return ret;
    }

    // 338. 比特位计数
    public int[] countBits(int n) {
        int[] ret = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            ret[i] = ret[i & (i - 1)] + 1;
        }
        return ret;
    }

    /**
     * 双指针
     */
    // 167. 两数之和2-输入有序数组
    public int[] twoSumOrdered(int[] numbers, int target) {
        int lo = 0, hi = numbers.length - 1;
        while (lo < hi) {
            int sum = numbers[lo] + numbers[hi];
            if (sum == target) {
                return new int[] { lo + 1, hi + 1 };
            } else if (sum < target) {
                lo++;
            } else {
                hi--;
            }
        }
        return null;
    }

    // 633. 平方数之和
    public boolean judgeSquareSum(int c) {
        long lo = 0, hi = (long) Math.sqrt(c);
        while (lo <= hi) {
            long squareSum = lo * lo + hi * hi;
            if (squareSum == c) {
                return true;
            } else if (squareSum < c) {
                lo++;
            } else {
                hi--;
            }
        }
        return false;
    }

    // 345. 反转字符串中的元音字母
    public String reverseVowels(String s) {
        char[] cs = s.toCharArray();
        int i = 0, j = cs.length - 1;
        while (i < j) {
            while (!isVowel(cs[i]) && i < j)
                i++;
            while (!isVowel(cs[j]) && i < j)
                j--;
            if (i < j) {
                char t = cs[i];
                cs[i++] = cs[j];
                cs[j--] = t;
            }
        }
        return new String(cs);
    }

    private boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
                c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
    }

    // 680. 验证回文串2
    public boolean validPalindromeRemoveOne(String s) {
        int i = 0, j = s.length() - 1;
        while (s.charAt(i) == s.charAt(j) && i < j) {
            i++;
            j--;
        }
        if (i >= j)
            return true;
        if (s.charAt(i) == s.charAt(j - 1)) {
            int ti = i;
            int tj = j - 1;
            while (s.charAt(ti) == s.charAt(tj) && ti < tj) {
                ti++;
                tj--;
            }
            if (ti >= tj)
                return true;
        }
        if (s.charAt(i + 1) == s.charAt(j)) {
            i++;
        }
        while (s.charAt(i) == s.charAt(j) && i < j) {
            i++;
            j--;
        }
        return i >= j;
    }

    // 88. 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m + n - 1, i1 = m - 1, i2 = n - 1;
        while (i1 >= 0 && i2 >= 0) {
            if (nums2[i2] >= nums1[i1]) {
                nums1[i--] = nums2[i2--];
            } else {
                nums1[i--] = nums1[i1--];
            }
        }
        if (i2 >= 0) {
            System.arraycopy(nums2, 0, nums1, 0, i2 + 1);
        }
    }

    // 141. 环形链表
    public boolean hasCycle(ListNode head) {
        if (head == null)
            return false;
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            if (slow == fast)
                return true;
            slow = slow.next;
            fast = fast.next.next;
        }
        return false;
    }

    // 524. 通过删除字母匹配到字典里最长单词
    public String findLongestWord(String s, List<String> dictionary) {
        String longestWord = "";
        int len = s.length();
        for (String word : dictionary) {
            int longest = longestWord.length(), currLen = word.length();
            if (currLen < longest || (currLen == longest) && longestWord.compareTo(word) < 0) {
                continue;
            }
            int i = 0, j = 0;
            while (i < currLen && j < len) {
                if (word.charAt(i) == s.charAt(j++))
                    i++;
            }
            if (i == currLen)
                longestWord = word;
        }
        return longestWord;
    }

    /**
     * 排序
     */
    // 215. 数组中的第K个最大元素(快速选择)
    public int findKthLargest(int[] nums, int k) {
        k = nums.length - k;
        int l = 0, h = nums.length - 1;
        while (l < h) {
            int pivotIndex = randomPartition(nums, l, h);
            if (pivotIndex == k) {
                break;
            } else if (pivotIndex < k) {
                l = pivotIndex + 1;
            } else {
                h = pivotIndex - 1;
            }
        }
        return nums[k];
    }

    private int randomPartition(int[] nums, int l, int h) {
        int r = random.nextInt(h - l + 1) + l;
        swap(nums, l, r);
        return partition(nums, l, h);
    }

    private int partition(int[] nums, int l, int h) {
        int i = l, j = h + 1;
        while (true) {
            while (nums[++i] < nums[l] && i < h)
                ;
            while (nums[--j] > nums[l] && j > l)
                ;
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
        }
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    // 75. 颜色分类
    public void sortColors(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        int i = 0;
        while (i <= hi) {
            if (nums[i] == 2) {
                swap(nums, hi--, i);
            } else if (nums[i] == 0) {
                swap(nums, lo++, i++);
            } else {
                i++;
            }
        }
    }

    /**
     * 贪心思想
     */

    // 455. 分饼干
    public int findContentChildren(int[] g, int[] s) {
        int i = 0, j = 0;
        Arrays.sort(g);
        Arrays.sort(s);
        while (j < s.length && i < g.length) {
            if (g[i] <= s[j]) {
                i++;
            }
            j++;
        }
        return i;
    }

    // 435. 无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {
        int cntToRemove = 0;
        // Arrays.sort(intervals, (int[] i1, int[] i2) -> i1[1] - i2[1]);
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[1]));
        int bound = Integer.MIN_VALUE;
        for (int[] i : intervals) {
            System.out.println(i[0]);
            System.out.println(i[1]);
            if (i[0] >= bound) {
                bound = i[1];
            } else {
                cntToRemove++;
            }
        }
        return cntToRemove;
    }

    // 452. 用最少数量的箭引爆气球
    public int findMinArrowShotsSortLeft(int[][] points) {
        if (points.length == 0)
            return 0;
        int cnt = 1;
        Arrays.sort(points, Comparator.comparingInt(o -> o[0]));
        int shot = points[0][1];
        for (int[] i : points) {
            if (shot < i[0]) {
                cnt++;
                shot = i[1];
            } else if (i[1] < shot) {
                shot = i[1];
            }
        }
        return cnt;
    }

    public int findMinArrowShotsSortRight(int[][] points) {
        if (points.length == 0)
            return 0;
        int cnt = 1;
        Arrays.sort(points, Comparator.comparingInt(o -> o[1]));
        int shot = points[0][1];
        for (int i = 1; i < points.length; i++) {
            if (points[i][0] <= shot) {
                continue;
            }
            cnt++;
            shot = points[i][1];
        }
        return cnt;
    }

    // 406. 根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (a, b) -> a[0] == b[0] ? a[1] - b[1] : b[0] - a[0]);
        List<int[]> queue = new ArrayList<>(people.length);
        for (int[] p : people) {
            queue.add(p[1], p);
        }
        return queue.toArray(new int[queue.size()][]);
    }

    // 121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        if (prices.length == 0)
            return 0;
        int profit = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (min > prices[i])
                min = prices[i];
            else
                profit = Math.max(profit, prices[i] - min);
        }
        return profit;
    }

    // 122. 买卖股票的最佳时机2
    public int maxProfit2(int[] prices) {
        if (prices.length == 0)
            return 0;
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
    }

    // 605. 种花问题
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int len = flowerbed.length;
        int cnt = 0;
        for (int i = 0; i < len && cnt < n; i++) {
            if (flowerbed[i] == 1) {
                continue;
            }
            int prev = i == 0 ? 0 : flowerbed[i - 1];
            int next = i == len - 1 ? 0 : flowerbed[i + 1];
            if (prev == 0 && next == 0) {
                flowerbed[i] = 1;
                cnt++;
            }
        }
        return cnt >= n;
    }

    // 392. 判断子问题
    public boolean isSubsequence(String s, String t) {
        int si = 0, ti = 0;
        int slen = s.length(), tlen = t.length();
        while (si < slen && ti < tlen) {
            if (s.charAt(si) == t.charAt(ti++)) {
                si++;
            }
        }
        return si == slen;
    }

    // 665. 非递减数列
    public boolean checkPossibility(int[] nums) {
        int cnt = 0;
        for (int i = 1; i < nums.length && cnt < 2; i++) {
            if (nums[i] >= nums[i - 1]) {
                continue;
            }
            cnt++;
            if (i - 2 >= 0 && nums[i - 2] > nums[i]) {
                nums[i] = nums[i - 1];
            } else {
                nums[i - 1] = nums[i];
            }
        }
        return cnt < 2;
    }

    // 53. 最大子数组和
    public int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE;
        int sum = -1;
        for (int num : nums) {
            if (sum < 0) {
                sum = 0;
            }
            sum += num;
            if (sum > maxSum) {
                maxSum = sum;
            }
        }
        return maxSum;
    }

    // 763. 划分字母区间
    public List<Integer> partitionLabels(String s) {
        int[] lastIndexOfChar = new int[26];
        for (int i = 0; i < s.length(); i++) {
            lastIndexOfChar[s.charAt(i) - 'a'] = i;
        }
        List<Integer> partitions = new ArrayList<>();
        int lastIndex = 0;
        int lastEnd = -1;
        for (int i = 0; i < s.length(); i++) {
            if (lastIndexOfChar[s.charAt(i) - 'a'] > lastIndex) {
                lastIndex = lastIndexOfChar[s.charAt(i) - 'a'];
            }
            if (i == lastIndex) {
                partitions.add(i - lastEnd);
                lastEnd = i;
            }
        }
        return partitions;
    }

    /**
     * 排序
     */
    // 二分查找
    public int binarySearch(int[] nums, int key) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == key) {
                return mid;
            } else if (nums[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    // 69. x的平方根
    public int mySqrt(int x) {
        if (x < 2)
            return x;
        int lo = 1, hi = x / 2;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            int sqrt = x / mid;
            if (sqrt == mid) {
                return mid;
            } else if (sqrt > mid) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return hi;
    }

    // 744. 寻找比目标字母大的最小字母
    public char nextGreatestLetter(char[] letters, char target) {
        int lo = 0, hi = letters.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (letters[mid] <= target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return hi >= letters.length ? letters[0] : letters[hi];
    }

    // 540. 有序数组中的单一元素
    public int singleNonDuplicate(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            boolean isOdd = (mid % 2 == 1);
            boolean leftEqual = mid == 0 ? false : nums[mid] == nums[mid - 1];
            boolean rightEqual = mid == nums.length - 1 ? false : nums[mid] == nums[mid + 1];
            if (!leftEqual && !rightEqual) {
                return nums[mid];
            }
            if (leftEqual) {
                if (isOdd)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            } else {
                if (isOdd)
                    hi = mid - 1;
                else
                    lo = mid + 1;
            }
        }
        return nums[hi];
    }

    // 着重思考在target左右的判断、lo和hi的移动策略
    public int singleNonDuplicateConsise(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (mid % 2 == 1) {
                mid--;
            }
            if (nums[mid] == nums[mid + 1]) {
                lo = mid + 2;
            } else {
                hi = mid;
            }
        }
        return nums[hi];
    }

    // 278. 第一个错误的版本
    public int firstBadVersion(int n) {
        int lo = 1, hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (isBadVersion(mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return hi;
    }

    private boolean isBadVersion(int version) {
        return false;
    }

    // 153. 寻找旋转排序数组中的最小值
    public int findMin(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] <= nums[hi]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return nums[hi];
    }

    // 34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int firstPos = findFirst(nums, target);
        if (firstPos == nums.length || nums[firstPos] != target) {
            return new int[] { -1, -1 };
        }
        int lastPos = findFirst(nums, target + 1) - 1;
        return new int[] { firstPos, lastPos };

    }

    private int findFirst(int[] nums, int target) {
        int lo = 0, hi = nums.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    /**
     * 分治
     */
    // 241. 为运算表达式设计优先级
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> ret = new ArrayList<>();
        int len = expression.length();
        for (int i = 0; i < len; i++) {
            char c = expression.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                List<Integer> left = diffWaysToCompute(expression.substring(0, i));
                List<Integer> right = diffWaysToCompute(expression.substring(i + 1));
                for (int leftOp : left) {
                    for (int rightOp : right) {
                        switch (c) {
                            case '+':
                                ret.add(leftOp + rightOp);
                                break;
                            case '-':
                                ret.add(leftOp - rightOp);
                                break;
                            case '*':
                                ret.add(leftOp * rightOp);
                                break;
                        }
                    }
                }
            }
        }
        if (ret.size() == 0) {
            ret.add(Integer.parseInt(expression));
        }
        return ret;
    }

    // 95. 不同的二叉搜索树
    public List<TreeNode> generateTrees(int n) {
        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int lo, int hi) {
        List<TreeNode> ret = new ArrayList<>();
        for (int i = lo; i <= hi; i++) {
            List<TreeNode> left = generateTrees(lo, i - 1);
            List<TreeNode> right = generateTrees(i + 1, hi);
            for (TreeNode leftSub : left) {
                for (TreeNode rightSub : right) {
                    ret.add(new TreeNode(i, leftSub, rightSub));
                }
            }
        }
        if (ret.size() == 0) {
            ret.add(null);
        }
        return ret;
    }

    /**
     * 动态规划
     */
    // 斐波那契数列
    // 70. 爬楼梯
    public int climbStairs(int n) {
        if (n == 1)
            return 1;
        int prev2 = 1, prev1 = 1;
        int curr = 0;
        for (int i = 2; i <= n; i++) {
            curr = prev2 + prev1;
            prev2 = prev1;
            prev1 = curr;
        }
        return curr;
    }

    // 198. 打家劫舍
    public int rob(int[] nums) {
        int prev2 = 0, prev1 = 0;
        int curr = 0;
        for (int i = 1; i < nums.length; i++) {
            curr = Math.max(prev2 + nums[i], prev1);
            prev2 = prev1;
            prev1 = curr;
        }
        return curr;
    }

    // 213. 打家劫舍2
    public int rob2(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        int prev2 = 0, prev1 = 0;
        int currWithFirst = 0, currWithLast = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            currWithFirst = Math.max(prev2 + nums[i], prev1);
            prev2 = prev1;
            prev1 = currWithFirst;
        }
        prev2 = 0;
        prev1 = 0;
        for (int i = 1; i < nums.length; i++) {
            currWithLast = Math.max(prev2 + nums[i], prev1);
            prev2 = prev1;
            prev1 = currWithLast;
        }
        return Math.max(currWithFirst, currWithLast);
    }

    // 信件错排

    // 母牛生产

    // 矩阵路径
    // 64. 最小路径和
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] dp = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j == 0) {
                    dp[j] = dp[j] + grid[i][j];
                } else {
                    if (i == 0) {
                        dp[j] = dp[j - 1] + grid[i][j];
                    } else {
                        dp[j] = Math.min(dp[j - 1], dp[j]) + grid[i][j];
                    }
                }
            }
        }
        return dp[n - 1];
    }

    // 62. 不同路径
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] = dp[j - 1] + dp[j];
            }
        }
        return dp[n - 1];
    }

    // 数组区间
    // 303. 区域和检索-数组不可变
    // 详见 leetcode.ds.NumArray

    // 413. 等差数列划分
    public int numberOfArithmeticSlices(int[] nums) {
        if (nums.length <= 2)
            return 0;
        int[] dp = new int[nums.length];
        dp[0] = 0;
        dp[1] = 0;
        for (int i = 2; i < dp.length; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                dp[i] = dp[i - 1] + 1;
            } else {
                dp[i] = 0;
            }
        }
        int ret = 0;
        for (int i : dp) {
            ret += i;
        }
        return ret;
    }

    // 分割整数
    // 343. 整数拆分
    public int integerBreak(int n) {
        int dp[] = new int[n + 1];
        for (int i = 2; i <= n; i++) {
            dp[i - 1] = Math.max(dp[i - 1], i - 1);
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], dp[j] * (i - j));
                // dp[i] = Math.max(dp[i], Math.max(dp[j] * (i - j), j * (i - j)))
            }

        }
        return dp[n];
    }

    // 279. 完全平方数
    public int numSquares(int n) {
        int dp[] = new int[n + 1];
        for (int i = 1; i < dp.length; i++) {
            dp[i] = i;
            for (int j = 1; i - j * j >= 0; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    // 91. 解码方法
    public int numDecodings(String s) {
        if (s.charAt(0) == '0')
            return 0;
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        char prev = s.charAt(0), curr;
        for (int i = 2; i < n + 1; i++) {
            curr = s.charAt(i - 1);
            if (curr != '0') {
                dp[i] += dp[i - 1];
            }
            if ((prev == '1' || prev == '2') && (curr >= '0' && curr <= '6')) {
                dp[i] += dp[i - 2];
            } else if (prev == '1' && (curr >= '7')) {
                dp[i] += dp[i - 2];
            }
            prev = curr;
        }
        return dp[n];
    }

    // 最长递增子序列
    // 300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int maxLength = 1;
        for (int i = 0; i < dp.length; i++) {
            int max = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    max = Math.max(max, dp[j] + 1);
                }
            }
            dp[i] = max;
            maxLength = Math.max(maxLength, max);
        }
        return maxLength;
    }

    public int lengthOfLISTails(int[] nums) {
        int len = 0;
        int[] tails = new int[nums.length];
        for (int num : nums) {
            int index = lengthOfLISTailsBinarySearch(tails, num, 0, len);;
            if (index == len) {
                len++;
            }
            tails[index] = num;
        }
        return len;
    }

    private int lengthOfLISTailsBinarySearch(int[] nums, int target, int lo, int hi) {
        int mid = 0;
        while (lo < hi) {
            mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    // 646. 最长数对链
    public int findLongestChain(int[][] pairs) {
        int[] dp = new int[pairs.length];
        int maxLength = 1;
        Arrays.sort(pairs, (a, b) -> a[0] - b[0]);
        for (int i = 0; i < dp.length; i++) {
            int max = 1;
            for (int j = 0; j < i; j++) {
                if (pairs[j][1] < pairs[i][0]) {
                    max = Math.max(max, dp[j] + 1);
                }
            }
            dp[i] = max;
            maxLength = Math.max(maxLength, max);
        }
        return maxLength;
    }

    // 376. 摆动序列
    public int wiggleMaxLength(int[] nums) {
        int[] dpPos = new int[nums.length];
        int[] dpNeg = new int[nums.length];
        int maxLength = 1;
        for (int i = 0; i < nums.length; i++) {
            int maxPos = 1, maxNeg = 1;
            for (int j = 0; j < i; j++) {
                int diff = nums[i] - nums[j];
                if (diff < 0) {
                    maxNeg = Math.max(maxNeg, dpPos[j] + 1);
                } else if (diff > 0) {
                    maxPos = Math.max(maxPos, dpNeg[j] + 1);
                }
            }
            dpPos[i] = maxPos;
            dpNeg[i] = maxNeg;
            maxLength = Math.max(maxLength, Math.max(maxPos, maxNeg));
        }
        return maxLength;
    }

    public int wiggleMaxLengthOn(int[] nums) {
        int up = 1, down = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                up = down + 1;
            } else if (nums[i] < nums[i - 1]) {
                down = up + 1;
            }
        }
        return Math.max(up, down);
    }

    // 最长公共子序列
    // 1143. 最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    // 0-1背包
    // 经典0-1背包
    public int knapsack(int W, int N, int[] weights, int[] values) {
        int[][] dp = new int[N + 1][W + 1];
        for (int i = 1; i <= N; i++) {
            int w = weights[i - 1], v = values[i - 1];
            for (int j = 1; j <= W; j++) {
                if (j >= w) {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - w] + v);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[N][W];
    }

    // 空间优化
    // 防止dp[i][j-w]覆盖dp[i-1][j-w]，倒序求解
    public int knapsack1D(int W, int N, int[] weights, int[] values) {
        int[] dp = new int[W + 1];
        for (int i = 1; i <= N; i++) {
            int w = weights[i - 1], v = values[i - 1];
            for (int j = W; j >= 1; j--) {
                if (j >= w) {
                    dp[j] = Math.max(dp[j], dp[j - w] + v);
                }
            }
        }
        return dp[W];
    }

    // 416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 == 1) {
            return false;
        }
        sum /= 2;
        boolean[] dp = new boolean[sum + 1];
        dp[0] = true;
        for (int num : nums) {
            for (int j = sum; j >= 1; j--) {
                if (j >= num) {
                    dp[j] = dp[j] || dp[j - num];
                }
            }
        }
        return dp[sum];
    }

    // 494. 目标和
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum + target < 0) return 0;
        if ((sum + target) % 2 == 1) return 0;
        int W = (sum + target) / 2;
        int[] dp = new int[W + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int j = W; j >= num; j--) {
                dp[j] += dp[j - num];
            }
        }
        return dp[W];
    }

    // DFS解法
    public int findTargetSumWaysDFS(int[] nums, int S) {
        return findTargetSumWaysDFS(nums, 0, S);
    }
    
    private int findTargetSumWaysDFS(int[] nums, int start, int S) {
        if (start == nums.length) {
            return S == 0 ? 1 : 0;
        }
        return findTargetSumWaysDFS(nums, start + 1, S + nums[start])
                + findTargetSumWaysDFS(nums, start + 1, S - nums[start]);
    }

    // 474. 一和零
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (String str : strs) {
            int len = str.length();
            int zeroCnt = 0, oneCnt = 0;
            for (int i = 0; i < len; i++) {
                if (str.charAt(i) == '0') {
                    zeroCnt += 1;
                }
            }
            oneCnt = len - zeroCnt;
            for (int i = m; i >= zeroCnt; i--) {
                for (int j = n; j >= oneCnt; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeroCnt][j - oneCnt] + 1);
                }
            }
        }
        return dp[m][n];
    }

    // 322. 找零钱的最少硬币数
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) return 0;
        int[] dp = new int[amount + 1];
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                if (i == coin) {
                    dp[i] = 1;
                } else if (dp[i] == 0 && dp[i - coin] != 0) {
                    dp[i] = dp[i - coin] + 1;
                } else if (dp[i - coin] != 0) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == 0 ? -1 : dp[amount];
    }

    // 2020年的精悍写法
    public int coinChange2020(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;

        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }

        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    // 518. 找零钱的硬币数组合
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }

    // 139. 字符串按单词列表分割
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (String word : wordDict) {
                int len = word.length();
                if (i >= len && word.equals(s.substring(i - len, i))) {
                    dp[i] = dp[i] || dp[i - len];
                }                
            }
        }
        return dp[n];
    }

    // 377. 组合总和
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (i >= num) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    // 股票交易
    // 309. 需要冷却期的股票交易
    public int maxProfitCooldown(int[] prices) {
        int n = prices.length;
        int[] buy = new int[n];
        int[] holdPos = new int[n];
        int[] sell = new int[n];
        int[] shortPos = new int[n];
        holdPos[0] = buy[0] = -prices[0];
        sell[0] = shortPos[0] = 0;
        for (int i = 1; i < n; i++) {
            buy[i] = shortPos[i - 1] - prices[i];
            holdPos[i] = Math.max(holdPos[i - 1], buy[i - 1]);
            sell[i] = Math.max(holdPos[i - 1], buy[i - 1]) + prices[i];
            shortPos[i] = Math.max(shortPos[i - 1], sell[i - 1]);
        }
        return Math.max(sell[n - 1], shortPos[n - 1]);
    }

    // 714. 需要交易费用的股票交易
    public int maxProfitFee(int[] prices, int fee) {
        int n = prices.length;
        int[] buy = new int[n];
        int[] holdPos = new int[n];
        int[] sell = new int[n];
        int[] shortPos = new int[n];
        holdPos[0] = buy[0] = -prices[0];
        sell[0] = shortPos[0] = 0;
        for (int i = 1; i < n; i++) {
            buy[i] = Math.max(shortPos[i - 1], sell[i - 1]) - prices[i];
            holdPos[i] = Math.max(holdPos[i - 1], buy[i - 1]);
            sell[i] = Math.max(holdPos[i - 1], buy[i - 1]) + prices[i] - fee;
            shortPos[i] = Math.max(shortPos[i - 1], sell[i - 1]);
        }
        return Math.max(sell[n - 1], shortPos[n - 1]);
    }

    // 123. 只能进行两次的股票交易
    public int maxProfitTwice(int[] prices) {
        int firstBuy = Integer.MIN_VALUE, firstSell = 0;
        int secondBuy = Integer.MIN_VALUE, secondSell = 0;
        for (int price : prices) {
            firstBuy = Math.max(firstBuy, -price);
            firstSell = Math.max(firstSell, firstBuy + price);
            secondBuy = Math.max(secondBuy, firstSell - price);
            secondSell = Math.max(secondSell, secondBuy + price);
        }
        return secondSell;
    }

    // 188. 只能进行k次的股票交易
    public int maxProfitKTimes(int k, int[] prices) {
        int[] buy = new int[k + 1];
        int[] sell = new int[k + 1];
        Arrays.fill(buy, Integer.MIN_VALUE);
        for (int price : prices) {
            for (int i = 1; i <= k; i++) {
                buy[i] = Math.max(buy[i], sell[i - 1] - price);
                sell[i] = Math.max(sell[i], buy[i] + price);
            }
        }
        return sell[k];
    }

    // 字符串编辑
    // 583. 删除两个字符串的字符使它们相等
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }

    // 转换为最长公共子序列问题
    public int minDistanceTransfer(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return m + n - 2 * dp[m][n];
    }

    // 72. 编辑距离
    public int minDistanceUpdate(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
                }
            }
        }
        return dp[m][n];
    }

    // 650. 复制粘贴字符
    public int minSteps(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[1] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; i + i * j <= n; j++) {
                int t = i + i * j;
                dp[t] = Math.min(dp[t], dp[i] + j + 1);
            }
        }
        return dp[n];
    }

    // 递归
    public int minStepsRecur(int n) {
        if (n == 1) return 0;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) return i + minSteps(n / i);
        }
        return n;
    }

    // 更优写法
    public int minStepsBetter(int n) {
        int[] dp = new int[n + 1];
        for (int i = 2; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                if (i % j == 0) {
                    dp[i] = Math.min(dp[i], dp[j] + i / j);
                    dp[i] = Math.min(dp[i], dp[i / j] + j);
                }
            }
        }
        return dp[n];
    }

    /**
     * 搜索
     */
    // BFS
    // 1091. 计算在网格中从原点到特定点的最短路径长度
    public int shortestPathBinaryMatrix(int[][] grids) {
        if (grids[0][0] == 1) return -1;
        int n = grids.length;
        int[][] direction = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
        boolean[][] visited = new boolean[n][n];
        int[][] depth = new int[n][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0, 0});
        visited[0][0] = true;
        int currDepth = 0;
        while (!queue.isEmpty()) {
            currDepth++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] curr = queue.poll();
                int x = curr[0], y = curr[1];
                depth[x][y] = currDepth;
                for (int[] d : direction) {
                    int tx = x + d[0], ty = y + d[1];
                    if (tx < 0 || tx >= n || ty < 0 || ty >= n) continue;
                    if (grids[tx][ty] == 1) continue;
                    if (visited[tx][ty]) continue;
                    queue.offer(new int[]{tx, ty});
                    visited[tx][ty] = true;
                }
            }
        }
        return depth[n - 1][n - 1] == 0 ? -1 : depth[n - 1][n - 1];
    }
    
    // 更普遍的BFS写法（为了普遍性用了visited数组，用了currDepth，用了depth数组）
    public int shortestPathBinaryMatrixGeneral(int[][] grids) {
        if (grids[0][0] == 1) {
            return -1;
        }
        int[][] direction = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
        int n = grids.length;
        int[][] depth = new int[n][n];
        boolean[][] visited = new boolean[n][n];
        Queue<int[]> queue = new LinkedList<>();


        queue.offer(new int[]{0, 0});
        visited[0][0] = true;
        depth[0][0] = 1;

        int currDepth = 1;
        while (!queue.isEmpty()) {
            int[] curr = queue.poll();
            int x = curr[0], y = curr[1];
            depth[x][y] = currDepth;
            if (x == n - 1 && y == n - 1) {
                return currDepth;
            }
            for (int[] d : direction) {
                int tx = x + d[0], ty = y + d[1];
                if (tx < 0 || tx >= n || ty < 0 || ty >= n) continue;
                if (grids[tx][ty] == 1 || visited[tx][ty]) continue;

                queue.offer(new int[]{tx, ty});
                visited[tx][ty] = true;
                depth[x][y] = currDepth + 1;
            }
            currDepth++;
        }
        return -1;
    }
    
    // 更好的题目写法（不用depth数组，不用visited数组）
    public int shortestPathBinaryMatrixBetter(int[][] grids) {
        if (grids[0][0] == 1) return -1;
        int[][] direction = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
        int n = grids.length;
        Queue<int[]> queue = new LinkedList<>();
        
        queue.offer(new int[]{0, 0});
        grids[0][0] = 1;
        int currDepth = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            currDepth++;
            for (int i = 0; i < size; i++) {
                int[] curr = queue.poll();
                int x = curr[0], y = curr[1];
                if (x == n - 1 && y == n - 1) {
                    return currDepth;
                }
                for (int[] d : direction) {
                    int nx = x + d[0], ny = y + d[1];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                    if (grids[nx][ny] == 1) continue;

                    queue.offer(new int[]{nx, ny});
                    grids[nx][ny] = 1;
                }
            }
        }
        return -1;
    }

    // 279. 组成整数的最小平方数数量


    // 127. 最短单词路径


    // DFS
    // 695. 查找最大的连通面积
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        maxAreaOfIslandVisited = new boolean[m][n];
        int maxArea = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxArea = Math.max(maxArea, maxAreaOfIslandDfs(grid, i, j));
            }
        }
        return maxArea;
    }

    private int[][] maxAreaOfIslandDirection = new int[][]{{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
    private boolean[][] maxAreaOfIslandVisited = null;

    private int maxAreaOfIslandDfs(int[][] grid, int x, int y) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length) return 0;
        if (grid[x][y] == 0) return 0;
        if (maxAreaOfIslandVisited[x][y]) return 0;
        maxAreaOfIslandVisited[x][y] = true;
        int area = 1;
        for (int[] d : maxAreaOfIslandDirection) {
            int nx = x + d[0], ny = y + d[1];
            area += maxAreaOfIslandDfs(grid, nx, ny);
        }
        return area;
    }

    // 200. 矩阵中的连通分量数目


    // 547. 好友关系的连通分量数目


    // 130. 填充封闭区域


    // 417. 能到达的太平洋和大西洋的区域


    // Backtracking（回溯）
    // 17. 数字键盘组合
    private String[] letterCombinationKeys = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    public List<String> letterCombination(String digits) {
        List<String> combinations = new ArrayList<>();
        letterCombinationBacktracking(new StringBuilder(), combinations, digits);
        return combinations;
        
    }

    private void letterCombinationBacktracking(StringBuilder prefix, List<String> combinations, String digits) {
        if (prefix.length() == digits.length()) {
            combinations.add(prefix.toString());
            return;
        }
        char currDigit = digits.charAt(prefix.length());
        String letters = letterCombinationKeys[currDigit - '0'];
        int len = letters.length();
        for (int i = 0; i < len; i++) {
            char c = letters.charAt(i);
            prefix.append(c);
            letterCombinationBacktracking(prefix, combinations, digits);
            prefix.deleteCharAt(prefix.length() - 1);
        }
    }

    // 93. IP地址划分


    // 79. 在矩阵中寻找字符串
    private int[][] existDirection = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    public boolean exist(char[][] board, String word) {
        int m = board.length, n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (existBacktracking(0, i, j, visited, board, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean existBacktracking(int k, int x, int y, boolean[][] visited, char[][] board, String word) {
        if (k == word.length()) {
            return true;
        }
        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length || visited[x][y]) {
            return false;
        }
        if (board[x][y] != word.charAt(k)) {
            return false;
        }
        visited[x][y] = true;
        for (int[] d : existDirection) {
            if (existBacktracking(k + 1, x + d[0], y + d[1], visited, board, word)) {
                return true;
            }
        }
        visited[x][y] = false;
        return false;
    }

    // 257.输出二叉树中所有从根到叶子的路径


    // 46. 排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> permutes = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        permuteBacktracking(new ArrayList<>(), permutes, visited, nums);
        return permutes;
    }

    public void permuteBacktracking(List<Integer> prefix, List<List<Integer>> permutes, boolean[] visited, int[] nums) {
        if (prefix.size() == nums.length) {
            permutes.add(new ArrayList<>(prefix));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) continue;
            visited[i] = true;
            prefix.add(nums[i]);
            permuteBacktracking(prefix, permutes, visited, nums);
            prefix.remove(prefix.size() - 1);
            visited[i] = false;
        }
    }

    // 47. 含有相同元素求排列
    // 相同元素的不同排列算一个，做法是，先将所有元素排序，在添加一个元素时先判断前一个元素是否和这个元素相等，如果相等且前一个元素未访问，就跳过

    // 77. 组合
    // 一是组合不在乎顺序了，不同顺序算一种，所以引入start，保持前后顺序，这样就不会重复
    // 二是因为要保持前后顺序所以长度为k的序列不可能从超过n-k+1之后的位置开始
    // 因为递归一定会start = i + 1，确保了一个元素不会用两次，所以不需要visited
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> combines = new ArrayList<>();
        combineBacktracking(new ArrayList<>(), combines, 1, n, k);
        return combines;
    }

    private void combineBacktracking(List<Integer> prefix, List<List<Integer>> combines, int start, int n, int k) {
        if (k == 0) {
            combines.add(new ArrayList<>(prefix));
            return;
        }
        for (int i = start; i <= n - k + 1; i++) {
            prefix.add(i);
            combineBacktracking(prefix, combines, i + 1, n, k - 1);
            prefix.remove(prefix.size() - 1);
        }
    }

    // 39. 组合求和
    // 是求组合，所以要用start，每个数字可以随便用，所以递归的时候不用传入start=i+1，只用传入start=i，也不用visited

    // 40. 含有相同元素的组合求和
    // 是求组合，所以要用start，每个元素只能用一次，所以递归要传start=i+1，因为每个元素只能用一次，所以本来不需要visited
    // 但是为了不让相同元素重复，需要用visited数组来作条件判断 candidates[i] == candidates[i - 1] && !hasVisited[i - 1] 则跳过

    // 216. 1-9数字的组合求和
    // 是求组合，所以要用start，每个元素只能用一次，所以递归要传start=i+1，因为每个元素只能用一次，所以不需要visited
    // 长度为k，所以可以剪枝，长度为k的序列，不可能从9-k+1之后的数字开始
    // 传入n和k如果都为0，说明找到了，如果只有n为0，说明k不够，直接返回，如果只有k为0，说明和没到，直接返回

    // 78. 子集
    // 相当于k从0到nums.length都求一次组合
    // 在函数主体中对k作一次迭代，用start，在递归时传入start = i + 1，不用visited，剪枝
    
    // 90. 含有相同元素求子集
    // 相当于k从0到nums.length都求一次组合
    // 有重复元素，所以要先排序，且要用visited数组，判断 candidates[i] == candidates[i - 1] && !hasVisited[i - 1] 则跳过
    // 在函数主体中对k作一次迭代，用start，在递归时传入start = i + 1，不用visited，剪枝

    // 131. 分割字符串使得每个部分都是回文数
    // 在判断子串是回文串才加入到prefix中

    // 37. 数独
    // 首先对行列和各都创建每个数字的bool数组，这样就能最快地判断这个行列格哪些数字没被占用
    // 其次是对网格的回溯，如果是有数字的直接开头就continue，然后对这个网格1-9能下的都试一下
    // 递归传参数时，不加1，而是在每个递归的开头判断下一个需要下的格子

    // 51. N皇后
    // 略

    /**
     * 数学
     */
    // 204. 生成素数序列
    public int countPrimes(int n) {
        boolean[] notPrimes = new boolean[n + 1];
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (notPrimes[i]) {
                continue;
            }
            count++;
            for (long j = (long)(i) * i; j < n; j += i) {
                notPrimes[(int) j] = true;
            }
        }
        return count;
    }
    
    // 最大公约数和最小公倍数
    // 最大公约数
    public int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // 最小公倍数
    public int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    // 进制转换
    // 504. 7进制
    public String convertToBase7(int num) {
        if (num == 0) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        boolean isNegative = num < 0;
        if (isNegative) {
            num = -num;
        }
        while (num > 0) {
            sb.append(num % 7);
            num /= 7;
        }
        String ret = sb.reverse().toString();
        return isNegative ? "-" + ret : ret;
    }

    public String convertToBase7Raw(int num) {
        return Integer.toString(num, 7);
    }
    
    // 405. 16进制
    public String toHex(int num) {
        char[] map = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
        if (num == 0) return "0";
        StringBuilder sb = new StringBuilder();
        while (num != 0) {
            sb.append(map[num & 0b1111]);
            num >>>= 4;
        }
        return sb.reverse().toString()
;    }

    // 168. 26进制
    public String convertToTitle(int n) {
        if (n == 0) {
            return "";
        }
        n--;
        return convertToTitle(n / 26) + (char) (n % 26 + 'A');
    }

    // 阶乘
    // 172. 统计阶乘尾部有多少个0
    public int trailingZeroes(int n) {
        return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5);
    }

    // 字符串加法减法
    // 67. 二进制加法
    public String addBinary(String a, String b) {
        int i = a.length() - 1, j = b.length() - 1, carry = 0;
        StringBuilder str = new StringBuilder();
        while (carry == 1 || i >= 0 || j >= 0) {
            if (i >= 0 && a.charAt(i--) == '1') {
                carry++;
            }
            if (j >= 0 && b.charAt(j--) == '1') {
                carry++;
            }
            str.append(carry % 2);
            carry /= 2;
        }
        return str.reverse().toString();
    }

    // 415. 字符串加法
    public String addStrings(String num1, String num2) {
        StringBuilder str = new StringBuilder();
        int carry = 0, i = num1.length() - 1, j = num2.length() - 1;
        while (carry == 1 || i >= 0 || j >= 0) {
            int x = i < 0 ? 0 : num1.charAt(i--) - '0';
            int y = j < 0 ? 0 : num2.charAt(j--) - '0';
            str.append((x + y + carry) % 10);
            carry = (x + y + carry) / 10;
        }
        return str.reverse().toString();
    }

    // 相遇问题
    // 462. 改变数组元素使所有的数组元素都相等
    // 排序或用快速选择找到中位数

    // 多数投票问题
    // 169. 数组中出现次数多于n/2的元素
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    // Boyer-Moore Majority Vote
    public int majorityElementMV(int[] nums) {
        int cnt = 0, majority = nums[0];
        for (int num : nums) {
            majority = (cnt == 0) ? num : majority;
            cnt = (majority == num) ? cnt + 1 : cnt - 1;
        }
        return majority;
    }

    // 其他
    // 367. 平方数
    public boolean isPerfectSquare(int num) {
        int subNum = 1;
        while (num > 0) {
            num -= subNum;
            subNum += 2;
        }
        return num == 0;
    }

    // 3的n次方
    public boolean isPowerOfThree(int n) {
        return n > 0 && (1162261467 % n == 0);
    }

    // 238. 乘积数组
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] products = new int[n];
        Arrays.fill(products, 1);
        int left = 1;
        for (int i = 1; i < n; i++) {
            left *= nums[i - 1];
            products[i] *= left;
        }
        int right = 1;
        for (int i = n - 2; i >= 0; i--) {
            right *= nums[i + 1];
            products[i] *= right;
        }
        return products;
    }

    // 628. 找出数组中的乘积最大的三个数
    public int maximumProduct(int[] nums) {
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE, min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        for (int n : nums) {
            if (n > max1) {
                max3 = max2;
                max2 = max1;
                max1 = n;
            } else if (n > max2) {
                max3 = max2;
                max2 = n;
            } else if (n > max3) {
                max3 = n;
            }
    
            if (n < min1) {
                min2 = min1;
                min1 = n;
            } else if (n < min2) {
                min2 = n;
            }
        }
        return Math.max(max1*max2*max3, max1*min1*min2);
    }
}
