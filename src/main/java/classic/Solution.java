package classic;

import java.util.Random;

public class Solution {
    // 字符串匹配kmp算法
    public static boolean kmp(String s, String t) {
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

    private static int[] buildNext(String t) {
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

    // 计算组合数的两种做法，推荐用乘法递推式
    public static long combinationAdd(int n, int m) {
        long[][] dp = new long[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= m && j <= i; j++) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }
        return dp[n][m];
    }

    public static long combinationProduct(int n, int m) {
        long ret = 1;
        for (int i = m + 1; i <= n; i++) {
            ret = ret * i / (i - m);
        }
        return ret;
    }


    // 常用模数10^9 + 7 = 1000000007
    // 因为该模数小于 Long.MAX_VALUE 开方，所以在以下跟模相关的运算中，没有对操作数取模，只对结果取模
    // 操作数是否需要取模，需要根据在给定模数下，操作数加法乘法是否可能溢出具体判断
    private static long MOD = 1000000007;

    // 模和
    public static long add(long x, long y) {
        return (x + y) % MOD;
    }

    // 模乘
    public static long mul(long x, long y) {
        return (x * y) % MOD;
    }

    // 快速幂
    public static long quickPow(long x, long n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        long temp = quickPow(x, n >> 1);
        return (n & 1) == 0 ? mul(temp, temp) : mul(x, mul(temp, temp));
    }

    // 分数模（小费马定理）
    // 分子a，分母b，求(a / b) mod p，其中p为素数
    // 小费马定理：a^(p - 1) mod p = 1 mod p
    // 可推出 a * b^(-1) mod p = a * b^(p - 2) mod p
    public static long fractionMod(long a, long b) {
        return mul(a, quickPow(b, MOD - 2));
    }

    // 二分查找
    public static int binarySearch(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    // 二分查找变体（寻找第一个target，如果不存在就返回比target大的第一个）
    public static int findFirst1(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }
    public static int findFirst2(int[] nums, int target) {
        int lo = 0, hi = nums.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    // 快速选择
    public static int quickSelect(int[] nums, int k) {
        int n = nums.length;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int pivotIndex = partition(nums, lo, hi);
            if (pivotIndex == k) {
                return nums[k];
            } else if (pivotIndex < k) {
                lo = pivotIndex + 1;
            } else {
                hi = pivotIndex - 1;
            }
        }
        return nums[k];
    }

    private static int partition(int[] nums, int lo, int hi) {
        int i = lo, j = hi + 1;
        while (true) {
            while (nums[++i] < nums[lo] && i < hi);
            while (nums[--j] > nums[lo] && j > lo);
            if (i >= j) break;
            swap(nums, i, j);
        }
        swap(nums, lo, j);
        return j;
    }

    private static void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    // 随机快速选择
    private static Random random;
    public static int randomizedSelect(int[] nums, int k) {
        int n = nums.length;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int pivotIndex = randomizedPartition(nums, lo, hi);
            if (pivotIndex == k) {
                return nums[k];
            } else if (pivotIndex < k) {
                lo = pivotIndex + 1;
            } else {
                hi = pivotIndex - 1;
            }
        }
        return nums[k];
    }
    private static int randomizedPartition(int[] nums, int lo, int hi) {
        int r = lo + random.nextInt(hi - lo + 1);
        swap(nums, r, lo);
        return partition(nums, lo, hi);
    }

    // 快速排序（优化partition）
    public static void quickSort(int[] nums) {
        quickSort(nums, 0, nums.length - 1);
    }

    private static void quickSort(int[] nums, int lo, int hi) {
        if (lo < hi) {
            int i = lo, j = hi + 1;
            while (true) {
                while (nums[++i] < nums[lo] && i < hi);
                while (nums[--j] > nums[lo] && j > lo);
                if (i >= j) break;
                swap(nums, i, j);
            }
            swap(nums, j, lo);
            quickSort(nums, lo, j - 1);
            quickSort(nums, j + 1, hi);
        }
    }

    public static void main(String[] args) {
        int[] nums = {0, 7, 3, 9, 2, 43, 101, 4};
        quickSort(nums);
        for (int i : nums) {
            System.out.println(i);
        }
    }
}
