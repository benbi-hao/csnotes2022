import leetcode.ds.MapSum;

public class Main {
    public static void main(String[] args){
        MapSum mapSum = new MapSum();
        mapSum.insert("apple", 3);  
        mapSum.sum("ap");           // 返回 3 (apple = 3)
        mapSum.insert("app", 2);    
        mapSum.sum("ap");           // 返回 5 (apple + app = 3 + 2 = 5)
    }
}