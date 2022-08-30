import leetcode.Solution;

public class Main {
    public static void main(String[] args){
        int[] nums = new int[]{0, 1, 0, 3, 12};
        int[] ret = new Solution().moveZeros(nums);
        for (int i : ret) {
            System.out.println(i);
        }
        
    }
}