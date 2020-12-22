//序列化 https://www.runoob.com/java/java-serialization.html

/*
请注意，一个类的对象要想序列化成功，必须满足两个条件：
该类必须实现 java.io.Serializable 接口。
该类的所有属性必须是可序列化的。如果有一个属性不是可序列化的，则该属性必须注明是短暂的。
*/

import java.io.*;

public class SerializeDemo {
    public static void main(String [] args)
    {
        SerializeEmployee e = new SerializeEmployee();
        e.name = "Reyan Ali";
        e.address = "Phokka Kuan, Ambehta Peer";
        e.SSN = 11122333;
        e.number = 101;
        try
        {
            FileOutputStream fileOut =
            new FileOutputStream("E:/Documents/Marvin/Source/Repos/BlueHope1987/StudyBase/helloJava/tmp/employee.ser");//调试环境相对路径会出错 需注意NTFS权限
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(e);
            out.close();
            fileOut.close();
            System.out.printf("Serialized data is saved in /tmp/employee.ser");
        }catch(IOException i)
        {
            i.printStackTrace();
        }
    }
}

