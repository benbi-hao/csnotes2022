package realexam.tencent;

import java.util.*;

public class Main {

    public static void main(String[] args) {
        countOfSamecheckSum();
    }

    // N次数据，n个数的数组，加上初始状态，每次去除掉一个下标，输出剩下的数组的中位数
    public static void floatingMedian() {
        Scanner in = new Scanner(System.in);
        int N = in.nextInt();
        for (int iN = 0; iN < N; iN++) {
            int n = in.nextInt();
            int[] nums = new int[n + 1];
            int[] toDels = new int[n - 1];
            for (int i = 1; i <= n; i++) {
                nums[i] = in.nextInt();
            }
            for (int i = 0; i < n - 1; i++) {
                toDels[i] = in.nextInt();
            }
            PriorityQueue<Integer> leftHeap = new PriorityQueue<>((i1, i2) -> i2 - i1);
            PriorityQueue<Integer> rightHeap = new PriorityQueue<>((i1, i2) -> i1 - i2);
            int leftSize = n / 2 + (n % 2 == 1 ? 1 : 0);
            int rightSize = n / 2;
            for (int i = 1; i <= n; i++) {
                leftHeap.offer(nums[i]);
                if (leftHeap.size() > leftSize) {
                    leftHeap.poll();
                }
                rightHeap.offer(nums[i]);
                if (rightHeap.size() > rightSize) {
                    rightHeap.poll();
                }
            }

            StringBuilder output = new StringBuilder();
            for (int i = 0; i < n - 1; i++) {
                int toDel = nums[toDels[i]];
                leftSize = leftHeap.size();
                rightSize = rightHeap.size();
                int leftTop = leftHeap.peek();
                int rightTop = rightHeap.peek();
                if (leftSize - 1 == rightSize) {
                    output.append(leftHeap.peek());
                } else if (leftSize == rightSize - 1) {
                    output.append(rightHeap.peek());
                } else {

                    output.append((leftTop + rightTop) / 2);
                    if ((leftTop + rightTop) % 2 == 1) {
                        output.append(".5");
                    }
                }
                output.append(' ');

                if (toDel <= leftTop) {
                    leftHeap.remove(toDel);
                    leftSize--;
                } else {
                    rightHeap.remove(toDel);
                    rightSize--;
                }

                int diff = leftSize - rightSize;
                if (diff > 1) {
                    rightHeap.offer(leftHeap.poll());
                } else if (diff < -1) {
                    leftHeap.offer(rightHeap.poll());
                }
            }

            if (!leftHeap.isEmpty()) {
                output.append(leftHeap.peek());
            } else {
                output.append(rightHeap.peek());
            }

            System.out.println(output.toString());
        }
    }

    // 01串，长度为n，子串长度为k，对每个长度为n的01串，所有长度为k子串的异或校验和与自己相等的字符串有多少个
    public static void countOfSamecheckSum() {
        Scanner in = new Scanner(System.in);
        int T = in.nextInt();
        for (int iT = 0; iT < T; iT++) {
            int n = in.nextInt();
            int k = in.nextInt();
            System.out.println(quickPow(2, n - k) - 1);
        }
    }

    private static long MOD = 1000000007;

    private static long mul(long a, long b) {
        return (a * b) % MOD;
    }

    private static long quickPow(long a, long n) {
        if (n == 0) return 1;
        if (n == 1) return a;
        long temp = quickPow(a, n >> 1);
        return (n & 1) == 0 ? mul(temp, temp) : mul(a, mul(temp, temp));
    }

    

}
