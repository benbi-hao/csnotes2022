import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args) {



        
    }

    // 49. 字母异位词分组
    // 将字母排序作为键
    public List<List<String>> groupAnagramsSort(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key, list);
        }
        List<List<String>> ret = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : map.entrySet()) {
            ret.add(entry.getValue());
        }
        
        return new ArrayList<>(map.values());
    }

    // 对字母频率编码作为键
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            int[] countMap = new int[26];
            for (char c : str.toCharArray()) {
                countMap[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 26; i++) {
                sb.append('a' + i);
                sb.append(countMap[i]);
            }
            String key = sb.toString();
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key, list);
        }

        return new ArrayList<>(map.values());
    }

    // 11. 乘最多水的容器
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int ret = 0;

        int lh = height[l], rh = height[r];
        while (l < r) {
            int currArea = Math.min(lh, rh) * (r - l);
            ret = Math.max(ret, currArea);
            if (lh < rh) {
                while (height[++l] <= lh && l < r);
                lh = height[l];
            } else {
                while (height[--r] <= rh && l < r);
                rh = height[r];
            }
        }
        return ret;
    }

    // 15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ret = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        if (nums[0] > 0 || nums[n - 1] < 0) return ret;
        for (int lo = 0; lo <= n - 3; lo++) {
            if (lo > 0 && nums[lo] == nums[lo - 1]) continue;
            int i = lo + 1, j = n - 1;
            int target = -nums[lo];
            while (i < j) {
                if (i > lo + 1 && nums[i] == nums[i - 1]) {
                    i++;
                    continue;
                }
                int sum = nums[i] + nums[j];
                if (sum == target) {
                    List<Integer> tuple = new ArrayList<>();
                    tuple.add(nums[lo]);
                    tuple.add(nums[i]);
                    tuple.add(nums[j]);
                    ret.add(tuple);
                    i++;
                    j--;
                } else if (sum < target) {
                    i++;
                } else {
                    j--;
                }
            }
        }
        return ret;
    }

    // 42.接雨水
    public int trap(int[] height) {
        int l = 0, r = height.length - 1;
        int lh = height[l], rh = height[r];
        int volume = 0;
        while (l < r) {
            if (lh < rh) {
                lh = Math.max(lh, height[++l]);
                volume += lh - height[l];
            } else {
                rh = Math.max(rh, height[--r]);
                volume += rh - height[r];
                
            }
        }
        return volume;
    }
}

