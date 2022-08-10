package leetcode.ds;

public class MapSum {
    private Node root;

    public MapSum() {
        root = new Node();
    }

    public void insert(String key, int val) {
        int len = key.length();
        Node curr = root;
        for (int i = 0; i < len; i++) {
            int idx = indexForChar(key.charAt(i));
            if (curr.children[idx] == null) {
                curr.children[idx] = new Node();
            }
            curr = curr.children[idx];
        }
        curr.val = val;
    }

    public int sum(String prefix) {
        int len = prefix.length();
        Node curr = root;
        for (int i = 0; i < len; i++) {
            int idx = indexForChar(prefix.charAt(i));
            if (curr.children[idx] == null) {
                return 0;
            }
            curr = curr.children[idx];
        }
        return sum(curr);
    }

    private int sum(Node root) {
        if (root == null) return 0;
        int sum = 0;
        for (Node child : root.children) {
            sum += sum(child);
        }
        return sum + root.val;
    }

    private int indexForChar(char c) {
        return c - 'a';
    }
}

class Node {
    int val;
    Node[] children;
    Node() {
        val = 0;
        children = new Node[26];
    }
}
