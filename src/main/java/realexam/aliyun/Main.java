package realexam.aliyun;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

    // 给一棵树，每个节点有正整数权值，计算路径权值乘积为6的路径的数量
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
        handleRecurrence(w, cntToParent, cntToCurr);
        ret += cntToCurr[6];
        cntToCurr[w] += 1;
        int[] cntBackCurr = new int[7];
        for (int child : graph[curr]) {
            if (child == parent) continue;
            int[] cntBackChild = dfs(curr, child);
            handleRecurrence(w, cntBackChild, cntBackCurr);
            handleRecurrence(w, cntBackChild, cntToCurr);
        }
        cntBackCurr[w] += 1;
        return cntBackCurr;
    }

    private static void handleRecurrence(int w, int[] from, int[] curr) {
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
}
