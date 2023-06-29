package leetcode.ds;

public class NumArray {
    private int leftSum[];

    public NumArray(int[] nums) {
        leftSum = new int[nums.length];
        leftSum[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            leftSum[i] = leftSum[i - 1] + nums[i];
        }
    }

    public int sumRange(int left, int right) {
        if (left == 0) return leftSum[right];
        return leftSum[right] - leftSum[left - 1];
    }
}
