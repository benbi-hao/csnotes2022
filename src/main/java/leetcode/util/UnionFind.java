package leetcode.util;

public class UnionFind {
    private int[] parent;
    
    public UnionFind(int n) {
        parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    public int find(int u) {
        if (parent[u] != u) {
            parent[u] = find(parent[u]);
        }
        return parent[u];
    }

    public void union(int u, int v) {
        parent[find(u)] = find(v);
    }

    public boolean isConnected(int u, int v) {
        return find(u) == find(v);
    }
}
