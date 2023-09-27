import java.util.*;

public class Main {


    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        weights = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            weights[i] = in.nextInt();
        }

        graph = new List[n + 1];
        for (int i = 1; i <= n; i++) {
            graph[i] = new ArrayList<>();
        }

        for (int i = 0; i < n - 1; i++) {
            int u = in.nextInt();
            int v = in.nextInt();
            graph[u].add(v);
            graph[v].add(u);
        }

        ret = 0;
        pathCount = new int[n + 1][7];
        dfs(0, 1);
        System.out.println(ret);
    }

    private static int ret;
    private static List<Integer>[] graph;
    private static int[] weights;
    private static int[][] pathCount;

    private static int[] dfs(int parent, int curr) {
        int[] cntToParent = pathCount[parent];
        int[] cntToCurr = pathCount[curr];
        int w = weights[curr];
        handle(w, cntToParent, cntToCurr);
        ret += cntToCurr[6];
        cntToCurr[w] += 1;
        int[] cntBackCurr = new int[7];
        for (int child : graph[curr]) {
            if (child == parent) continue;
            int[] cntBackChild = dfs(curr, child);
            handle(w, cntBackChild, cntBackCurr);
            handle(w, cntBackChild, cntToCurr);
        }
        cntBackCurr[w] += 1;
        return cntBackCurr;
    }

    private static void handle(int w, int[] from, int[] curr) {
        switch(w) {
            case 1:
                curr[1] += from[1];
                curr[2] += from[2];
                curr[3] += from[3];
                curr[6] += from[6];
                break;
            case 2:
                curr[2] += from[1];
                curr[6] += from[3];
                break;
            case 3:
                curr[3] += from[1];
                curr[6] += from[2];
                break;
            case 6:
                curr[6] += from[1];
                break;
            default:
                break;
        }
    }

    // public long combinationAdd(int n, int m) {
    //     long[][] dp = new long[n + 1][m + 1];
    //     for (int i = 0; i <= n; i++) {
    //         dp[i][0] = 1;
    //         for (int j = 1; j <= m && j <= i; j++) {
    //             dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
    //         }
    //     }
    //     return dp[n][m];
    // }

    // public long combinationProduct(int n, int m) {
    //     long ret = 1;
    //     for (int i = m + 1; i <= n; i++) {
    //         ret = ret * i / (i - m);
    //     }
    //     return ret;
    // }

    
}

