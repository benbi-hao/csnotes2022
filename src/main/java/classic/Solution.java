package classic;

public class Solution {
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
}
