//反序列化 读取SerializeDemo.java的序列化

import java.io.*;

public class SerializeRead {
    public static void main(String [] args)
    {
        SerializeEmployee e = null;
        try
        {
            FileInputStream fileIn = new FileInputStream("E:/Documents/Marvin/Source/Repos/BlueHope1987/StudyBase/helloJava/tmp/employee.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            e = (SerializeEmployee) in.readObject();
            in.close();
            fileIn.close();
        }catch(IOException i)
        {
            i.printStackTrace();
            return;
        }catch(ClassNotFoundException c)
        {
            System.out.println("Employee class not found");
            c.printStackTrace();
            return;
        }
        System.out.println("Deserialized Employee...");
        System.out.println("Name: " + e.name);
        System.out.println("Address: " + e.address);
        System.out.println("SSN: " + e.SSN);
        System.out.println("Number: " + e.number);
        }
}
