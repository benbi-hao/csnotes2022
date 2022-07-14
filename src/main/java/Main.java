import leetcode.ds.ListNode;
import leetcode.Solution;

public class Main {
    public static void main(String[] args){
        
        ListNode head = new ListNode(1, new ListNode(2, new ListNode(2, new ListNode(1))));

        System.out.println(new Solution().isPalindrome(head));
    }
}