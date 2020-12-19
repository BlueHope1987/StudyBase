//Java集合框架与算法 迭代器 遍历 ArrayList
// https://www.runoob.com/java/java-collections.html

import java.util.*;

public class IteratorTest {
    public static void main(String[] args) {
        ///////////////////////////////////////////
        //遍历 ArrayList https://www.runoob.com/java/java-arraylist.html
        //ArrayList 类是一个可以动态修改的数组，与普通数组的区别就是它是没有固定大小的限制，我们可以添加或删除元素。
        //方法：.add .get(0) .set(0,"") .remove(0) .size()
        //引用对象<String> <Boolean> <Byte> <Short> <Integer> <Long> <Float> <Double> <Character>

        List<String> list=new ArrayList<String>();
        list.add("Hello");
        list.add("World");
        list.add("HAHAHAHA");
        System.out.println(list);
        System.out.println(list.get(1));
        list.set(1,"World!");
        System.out.println(list.get(1));
        list.set(1,"World");
        System.out.println(list.remove(2));
        System.out.println(list.size());
        System.out.println(list);
        list.add("HAHAHAHA");
        System.out.println(list.size());
        System.out.println(list);
        //Collections.sort(list) 排序

        System.out.println("============ArrayList遍历方法===========");
        //第一种遍历方法使用 For-Each 遍历 List
        for (String str : list) {            //也可以改写 for(int i=0;i<list.size();i++) 这种形式
           System.out.println(str);
        }
    
        //第二种遍历，把链表变为数组相关的内容进行遍历
        String[] strArray=new String[list.size()];
        list.toArray(strArray);
        for(int i=0;i<strArray.length;i++) //这里也可以改写为  for(String str:strArray) 这种形式
        {
           System.out.println(strArray[i]);
        }
        
       //第三种遍历 使用迭代器进行相关遍历
        
        Iterator<String> ite=list.iterator();
        while(ite.hasNext())//判断下一个元素之后有值
        {
            System.out.println(ite.next());
        }

        ///////////////////////////////////////
        //遍历 Map
        System.out.println("============Map遍历方法===========");
        Map<String, String> map = new HashMap<String, String>();
        map.put("1", "value1");
        map.put("2", "value2");
        map.put("3", "value3");
        
        //第一种：普遍使用，二次取值
        System.out.println("通过Map.keySet遍历key和value：");
        for (String key : map.keySet()) {
        System.out.println("key= "+ key + " and value= " + map.get(key));
        }
        
        //第二种
        System.out.println("通过Map.entrySet使用iterator遍历key和value：");
        Iterator<Map.Entry<String, String>> it = map.entrySet().iterator();
        while (it.hasNext()) {
        Map.Entry<String, String> entry = it.next();
        System.out.println("key= " + entry.getKey() + " and value= " + entry.getValue());
        }
        
        //第三种：推荐，尤其是容量大时
        System.out.println("通过Map.entrySet遍历key和value");
        for (Map.Entry<String, String> entry : map.entrySet()) {
        System.out.println("key= " + entry.getKey() + " and value= " + entry.getValue());
        }
        
        //第四种
        System.out.println("通过Map.values()遍历所有的value，但不能遍历key");
        for (String v : map.values()) {
        System.out.println("value= " + v);
        }
    }
}
