package leetcode.util;

public class UnionFind {
    private int[] color;
    
    public UnionFind(int size) {
        color = new int[size + 1];
        for (int i = 1; i <= size; i++) {
            color[i] = i;
        }
    }

    public void union(int u, int v) {
        int colorV = color[v];
        if (color[u] == colorV) return;
        for (int i = 1; i < color.length; i++) {
            if (color[i] == colorV) color[i] = color[u];
        }
    }

    public boolean isConnected(int u, int v) {
        return color[u] == color[v];
    }
}
