import leetcode.Solution;

public class Main {
    public static void main(String[] args){
        // int[] array = new int[]{1, 0, 2, 3, 4};
        // int ret = new Solution().maxChunksToSorted(array);
        // System.out.println(ret);
        int[][] intervals = {{1, 2}, {2, 3}, {3, 4}, {1, 3}};
        new Solution().eraseOverlapIntervals(intervals);

    }
}