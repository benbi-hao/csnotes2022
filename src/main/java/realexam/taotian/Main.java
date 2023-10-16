package realexam.taotian;

import java.util.Scanner;

public class Main {
    // 给一个n*m矩阵，每个点两种颜色，R、W、?，？处可以涂色R或W，对每个2*2子矩阵，如果全是R则有1权值，问在所有涂色方法下所有可能的矩阵权值和是多少
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int m = in.nextInt();
        char[][] matrix = new char[n][m];
        long k = 0;
        for (int i = 0; i < n; i++) {
            matrix[i] = in.next().toCharArray();
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == '?') k++;
            }
        }
        
        long[] q2Cnt = new long[5];
        int most = Math.min(4, (int) k);
        for (int i = 0; i <= most; i++) {
            q2Cnt[i] = quickPow(2, k - i);
        }

        long total = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < m - 1; j++) {
                int qCnt = 0;
                if (matrix[i][j] == 'M') continue;
                else if (matrix[i][j] == '?') qCnt++;
                if (matrix[i][j + 1] == 'M') continue;
                else if (matrix[i][j + 1] == '?') qCnt++;
                if (matrix[i + 1][j] == 'M') continue;
                else if (matrix[i + 1][j] == '?') qCnt++;
                if (matrix[i + 1][j + 1] == 'M') continue;
                else if (matrix[i + 1][j + 1] == '?') qCnt++;
                total = add(total, q2Cnt[qCnt]);
            }
        }

        System.out.println(total);

    }

    private static long MOD = 1000000007;

    private static long add(long a, long b) {
        return (a + b) % MOD;
    }

    private static long mul(long a, long b) {
        return (a * b) % MOD;
    }

    private static long quickPow(long x, long n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        long temp = quickPow(x, n >> 1);
        return (n & 1) == 0 ? mul(temp, temp) : mul(x, mul(temp, temp));
    }
}
