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

        System.out.println("============ArrayList基础===========");
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
        
        Iterator<String> ite=list.iterator(); //Java Iterator（迭代器） https://www.runoob.com/java/java-iterator.html
        while(ite.hasNext())//判断下一个元素之后有值
        {
            System.out.println(ite.next());
        }

        ///////////////////////////////////////
        //遍历 LinkedList https://www.runoob.com/java/java-linkedlist.html
        //链表可分为单向链表和双向链表。Java LinkedList（链表） 类似于 ArrayList，是一种常用的数据容器。增加和删除对操作效率更高，而查找和修改的操作效率较低。
        System.out.println("============LinkedList基础===========");
        LinkedList<String> sites = new LinkedList<String>();
        sites.add("Google");
        sites.add("Runoob");
        sites.add("Taobao");
        sites.add("Weibo");
        System.out.println(sites);
        sites.addFirst("Wiki");
        System.out.println(sites);
        sites.addLast("Wiki2");
        System.out.println(sites);
        sites.removeFirst();
        System.out.println(sites);
        sites.removeLast();
        System.out.println(sites);
        System.out.println(sites.getFirst());
        System.out.println(sites.getLast());

        //迭代元素
        System.out.println("============LinkedList迭代方法===========");
        for (int size = sites.size(), i = 0; i < size; i++) {
            System.out.println(sites.get(i));
        }
        //for-each方法
        for (String i : sites) {
            System.out.println(i);
        }
        sites=null;
        ///////////////////////////////////////
        //遍历 Map https://www.runoob.com/java/java-hashmap.html
        //HashMap 是一个散列表，它存储的内容是键值对(key-value)映射。是无序的，即不会记录插入的顺序。
        //key 与 value 类型可以不同
        System.out.println("============HashMap基础===========");
        HashMap<Integer, String> Sites = new HashMap<Integer, String>();
        Sites.put(1, "Google");
        Sites.put(2, "Runoob");
        Sites.put(3, "Taobao");
        Sites.put(4, "Zhihu");
        System.out.println(Sites);
        System.out.println(Sites.get(3));
        Sites.remove(4);
        System.out.println(Sites);
        Sites.clear();
        System.out.println(Sites);
        Sites=null;

        HashMap<String, String> Sites2 = new HashMap<String, String>();
        Sites2.put("one", "Google");
        Sites2.put("two", "Runoob");
        Sites2.put("three", "Taobao");
        Sites2.put("four", "Zhihu");
        System.out.println(Sites2);
        System.out.println(Sites2.size());
        
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
        
        System.out.println("通过Map.values遍历value：");
        for (String value : map.values()) {
            System.out.println(value + ", ");;
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
