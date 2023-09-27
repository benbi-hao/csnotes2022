package classic;

public class Solution {
    // 字符串匹配kmp算法
    public boolean kmp(String s, String t) {
        int[] next = buildNext(t);
        int n = s.length();
        int m = t.length();
        int i = 0, j = 0;
        while (i < n && j < m) {
            if (s.charAt(i) == t.charAt(j)) {
                i++;
                j++;
            } else if (j != 0){
                j = next[j];
            } else {
                i++;
            }
        }
        return j == m;
    }

    private int[] buildNext(String t) {
        int len = t.length();
        int[] next = new int[len];
        int i = 1, curr = 0;
        while (i < len) {
            if (t.charAt(i) == t.charAt(curr)) {
                next[i++] = ++curr;
            } else if (curr != 0) {
                curr = next[curr - 1];
            } else {
                i++;
            }
        }
        return next;
    }

    // 计算组合数的两种做法，推荐用乘法递推式
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
