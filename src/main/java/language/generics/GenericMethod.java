package language.generics;

import java.util.ArrayList;

// 泛型类型参数的名字无所谓，不影响使用，只是按照规范不同名字有不同含义
// E: Element T: Type K: Key V: Value N: Number ?: 不确定
public class GenericMethod {
    // 实际上是从参数E[] array获取到类型E，在编译时会对
    public static <E> void printArray(E[] array) {
        for (E element : array) {
            System.out.println(element);
        }
    }

    public static <E> E getFirst(Object[] array) {
        return (E) array[0];
    }

    public static void main(String[] args) {
        Integer[] intArray = { 1, 2, 3, 4, 5 };
        Double[] doubleArray = { 1.1, 2.2, 3.3, 4.4 };
        // 两种调用方式，没有传入泛型参数的话，会根据参数自动获取，如果参数列表里没用到类型参数，那实际上编译器做不了类型安全检查
        // 泛型方法类型参数只在该方法有作用
        GenericMethod.<Integer>printArray(intArray);
        GenericMethod.printArray(doubleArray);

        // 泛型类向后兼容，可以不用类型参数进行初始化
        ArrayList list = new ArrayList();

        // 类型参数实际传入时机
        // 泛型类：实例化时
        // 泛型方法：调用时
        // 泛型接口：声明实现时，也就是用类实现这个接口时，在接口名后面指定类型参数

        // 类型参数通配符和上界下界使用位置
        // 1. 方法参数里，可以传入不确定的泛型
        // 2. 声明引用变量时，可以赋值为不确定的泛型
        // “extends上限可读不可写” “super下限可写不可读”

    }
}