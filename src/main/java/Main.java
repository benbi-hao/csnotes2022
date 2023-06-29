import java.util.Arrays;

import leetcode.Solution;

public class Main {
    public static void main(String[] args){
        // int[] array = new int[]{1, 0, 2, 3, 4};
        // int ret = new Solution().maxChunksToSorted(array);
        // System.out.println(ret);
        int[] dp = new int[3];
        Arrays.fill(dp, 1);
        for (int i = 0; i < dp.length; i++) {
            System.out.println(dp[i]);
        }

    }
}