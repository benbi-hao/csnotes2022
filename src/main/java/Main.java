import leetcode.Solution;

public class Main {
    public static void main(String[] args){
        int[] temperatures = {73,74,75,71,69,72,76,73};
        int[] ret = new Solution().dailyTemperaturesForward(temperatures);
        for (int i : ret) {
            System.out.println(i);
        }
    }
}