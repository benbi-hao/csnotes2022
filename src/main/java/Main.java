import leetcode.Solution;

public class Main {
    public static void main(String[] args){
        int[][] matrix = new int[][]{{1, 5, 9}, {10, 11, 13}, {12, 14, 15}};
        int ret = new Solution().kthSmallest(matrix, 8);
        System.out.println(ret);
    }
}