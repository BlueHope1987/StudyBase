//Java枚举 Enumeration接口 https://www.runoob.com/java/java-enumeration-interface.html

import java.util.Vector;
import java.util.Enumeration;
public class EnumerationTester {
    public static void main(String args[]){
        Enumeration<String> days; //枚举(泛型类https://www.runoob.com/java/java-generics.html )
        Vector<String> dayNames= new Vector<String>();
        dayNames.add("Sunday");
        dayNames.add("Monday");
        dayNames.add("Tuesday");
        dayNames.add("Wednesday");
        dayNames.add("Thursday");
        dayNames.add("Friday");
        dayNames.add("Saturday");
        days=dayNames.elements();
        while(days.hasMoreElements()){
            System.out.println(days.nextElement());
        }
    }
}
