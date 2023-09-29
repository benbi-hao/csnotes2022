package leetcode.ds;

public class Trie {
    private class Node {
        boolean isLeaf;
        Node[] children;
        Node() {
            isLeaf = false;
            children = new Node[26];
        }
    }

    private Node root;

    public Trie() {
        root = new Node();
    }

    public void insert(String word) {
        int len = word.length();
        Node curr = root;
        for (int i = 0; i < len; i++) {
            int index = indexForChar(word.charAt(i));
            if (curr.children[index] == null) {
                curr.children[index] = new Node();
            }
            curr = curr.children[index];
        }
        curr.isLeaf = true;
    }

    public boolean search(String word) {
        int len = word.length();
        Node curr = root;
        for (int i = 0; i < len; i++) {
            int index = indexForChar(word.charAt(i));
            if (curr.children[index] == null) {
                return false;
            }
            curr = curr.children[index];
        }
        return curr.isLeaf;
    }

    public boolean startsWith(String prefix) {
        int len = prefix.length();
        Node curr = root;
        for (int i = 0; i < len; i++) {
            int index = indexForChar(prefix.charAt(i));
            if (curr.children[index] == null) {
                return false;
            }
            curr = curr.children[index];
        }
        return true; 
    }

    private int indexForChar(char ch) {
        return ch - 'a';
    }
}

