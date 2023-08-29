

public class Main {
    public static void main(String[] args){
        // int[] array = new int[]{1, 0, 2, 3, 4};
        // int ret = new Solution().maxChunksToSorted(array);
        // System.out.println(ret);
        // int[] nums = {1, 2, 3, 4};
        Main o = new Main();
        long startTime1 = System.nanoTime();
        o.combinationAdd(8, 2);
        long endTime1 = System.nanoTime();
        System.out.println(endTime1 - startTime1);

        long startTime2 = System.nanoTime();
        o.combinationProduct(8, 2);
        long endTime2 = System.nanoTime();
        System.out.println(endTime2 - startTime2);
    }

    public long combinationAdd(int n, int m) {
        long[][] dp = new long[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= m && j <= i; j++) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }
        return dp[n][m];
    }

    public long combinationProduct(int n, int m) {
        long ret = 1;
        for (int i = m + 1; i <= n; i++) {
            ret = ret * i / (i - m);
        }
        return ret;
    }

    
}