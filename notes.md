java -1 % 5 = -1

java remove(int) 是按数组删除 remove(E) remove(Integer)是按值删除

java new arrayList<>(10) capacity为10，但是size为0

java set不支持随机取和随机删

java 快速判断完全平方数
```
private boolean isSquare(int a) {
    double b = Math.sqrt(a);
    return b - (int)b == 0;
}
```

java 创建临接表 用泛型数组更方便

java map方法 get put containsKey (containsValue不常用) getOrDefault clear remove

java map遍历方法，entrySet最快，迭代器和Foreach差不多

```
// ForEach entrySet 比keySet快一倍
for (Map.Entry<Integer, String> entry : map.entrySet()) {
    System.out.println(entry.getKey());
    System.out.println(entry.getValue());
}

// ForEach keySet
for (Integer key : map.keySet()) {
    System.out.println(key);
    System.out.println(map.get(key));
}

// Iterator entrySet
Iterator<Map.Entry<Integer, String>> iterator = map.entrySet().iterator();
while (iterator.hasNext()) {
    Map.Entry<Integer, String> entry = iterator.next();
    System.out.println(entry.getKey());
    System.out.println(entry.getValue());
}

// Iterator keyset
Iterator<Integer> iterator = map.keySet().iterator();
while (iterator.hasNext()) {
    Integer key = iterator.next();
    System.out.println(key);
    System.out.println(map.get(key));
}
```

java 打印2位小数点 System.out.printf("%.2f", a);

求排列数
```
public long permutation(int n, int m) {
    long ret = 1;
    for (long i = n; i > n - m; i--) {
        ret *= i;
    }
    return ret;
}
```

求组合数
```
// 用加法递推公式
public long combinationAdd(int n, int m) {
    long[][] dp = new long[n + 1][m + 1];
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 1;
        for (int j = 1; j <= m && j <= i; j++) {
            dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
        }
    }
    return dp[n][m];
}

// 用乘法递推公式（更快一些）
public long combinationProduct(int n, int m) {
    long ret = 1;
    for (int i = m + 1; i <= n; i++) {
        ret = ret * i / (i - m);
    }
    return ret;
}

```

java "." "|" "*" "+" 都是正则表达式转义字符，要加"\\"，如"\\."


shell 高级逻辑运算 && 和 || 要加 两个方括号 [[  ]]
shell 字符串运算符号 -z -n 的操作数 要加双引号 -z "$hello"
shell = == 都可以用来判断字符串相等和数字相等 -eq 用来判断数字相等
shell if then then要换行，在if的下一行，elif同理
shell 计算可以用 expr 也可以用 (())直接写表达式


mysql round(a, 2) 保留两位小数
mysql date_format(d, '%Y-%m-%d %h:%i:%s')