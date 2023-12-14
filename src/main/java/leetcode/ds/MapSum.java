package leetcode.ds;

public class MapSum {
    private MapSumNode root;

    public MapSum() {
        root = new MapSumNode();
    }

    public void insert(String key, int val) {
        int len = key.length();
        MapSumNode curr = root;
        for (int i = 0; i < len; i++) {
            int idx = indexForChar(key.charAt(i));
            if (curr.children[idx] == null) {
                curr.children[idx] = new MapSumNode();
            }
            curr = curr.children[idx];
        }
        curr.val = val;
    }

    public int sum(String prefix) {
        int len = prefix.length();
        MapSumNode curr = root;
        for (int i = 0; i < len; i++) {
            int idx = indexForChar(prefix.charAt(i));
            if (curr.children[idx] == null) {
                return 0;
            }
            curr = curr.children[idx];
        }
        return sum(curr);
    }

    private int sum(MapSumNode root) {
        if (root == null) return 0;
        int sum = 0;
        for (MapSumNode child : root.children) {
            sum += sum(child);
        }
        return sum + root.val;
    }

    private int indexForChar(char c) {
        return c - 'a';
    }
}

class MapSumNode {
    int val;
    MapSumNode[] children;
    MapSumNode() {
        val = 0;
        children = new MapSumNode[26];
    }
}
