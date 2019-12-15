#include <iostream>
#include <vector>
using namespace std;

class Shape
{
    //C++ 接口是使用抽象类来实现的，抽象类与数据抽象互不混淆，数据抽象是一个把实现细节与相关的数据分离开的概念。
    //如果类中至少有一个函数被声明为纯虚函数，则这个类就是抽象类。纯虚函数是通过在声明中使用 "= 0" 来指定的
    //https://www.runoob.com/cplusplus/cpp-interfaces.html

    public:
    //提供接口框架的纯虚函数
    virtual int getArea()=0;
    void setWidth(int w)
    {
        width=w;
    }
    void setHeight(int h)
    {
        height=h;
    }
    protected:
    int width;
    int height;
};
//派生类
class Rectangle:public Shape
{
    public:
    int getArea()
    {
        return (width*height);
    }
};
class Triangle:public Shape
{
    public:
    int getArea()
    {
        return(width*height)/2;
    }
};



int main()
{
    //接口教程
    Rectangle Rect;
    Triangle Tri;
    Rect.setWidth(5);
    Rect.setHeight(7);
    //输出对象的面积
    cout<<"Total Rectangle area: "<<Rect.getArea()<<endl;
    Tri.setWidth(5);
    Tri.setWidth(7);
    cout<<"Total Rectangle area: "<<Tri.getArea()<<endl;


    // 教程：https://www.runoob.com/cplusplus/cpp-stl-tutorial.html
    /*
    C++ STL（标准模板库）是一套功能强大的 C++ 模板类，提供了通用的模板类和函数，这些模板类和函数可以实现多种流行和常用的算法和数据结构，如向量、链表、队列、栈。
    C++ 标准模板库的核心包括以下三个组件：

    组件	描述
    容器（Containers）	容器是用来管理某一类对象的集合。C++ 提供了各种不同类型的容器，比如 deque、list、vector、map 等。
    算法（Algorithms）	算法作用于容器。它们提供了执行各种操作的方式，包括对容器内容执行初始化、排序、搜索和转换等操作。
    迭代器（iterators）	迭代器用于遍历对象集合的元素。这些集合可能是容器，也可能是容器的子集。
    
    这三个组件都带有丰富的预定义函数，帮助我们通过简单的方式处理复杂的任务。
    */
    //该程序演示了向量容器（一个 C++ 标准的模板），它与数组十分相似，唯一不同的是，向量在需要扩展大小的时候，会自动处理它自己的存储需求

    //创建一个向量存储int
    vector<int> vec;
    int i;

    //显示vec的原始大小
    cout << "vector size = " << vec.size() << endl;

    //推入5个值到向量中
    for (i = 0; i < 5; i++)
    {
        vec.push_back(i); //成员函数在向量的末尾插入值，如果有必要会扩展向量的大小。
    }

    //显示vec扩展后的大小
    cout << "extended vector size = " << vec.size() << endl;

    //访问向量中的5个值
    for (i = 0; i < 5; i++)
    {
        cout << "value of vec [" << i << "] = " << vec[i] << endl;
    }

    //使用迭代器iterator访问值
    vector<int>::iterator v = vec.begin();
    while (v != vec.end())
    {
        cout << "value of v = " << *v << endl;
        v++;
    }

    /*
    C++ STL 之 vector 的 capacity 和 size 属性区别
    size 是当前 vector 容器真实占用的大小，也就是容器当前拥有多少个容器。
    capacity 是指在发生 realloc 前能允许的最大元素数，即预分配的内存空间。
    当然，这两个属性分别对应两个方法：resize() 和 reserve()。
    使用 resize() 容器内的对象内存空间是真正存在的。
    使用 reserve() 仅仅只是修改了 capacity 的值，容器内的对象并没有真实的内存空间(空间是"野"的)。
    此时切记使用 [] 操作符访问容器内的对象，很可能出现数组越界的问题。
    */
    vector<int> v2;
    cout << "v.size() == " << v2.size() << " v.capacity() = " << v2.capacity() << endl;
    v2.reserve(10);
    cout << "v.size() == " << v2.size() << " v.capaciry() = " << v2.capacity() << endl;
    v2.resize(10);
    v2.push_back(0);
    cout << "v.size() == " << v2.size() << " v.capaciry() = " << v2.capacity() << endl;
    /*
    注： 对于 reserve(10) 后接着直接使用 [] 访问越界报错(内存是野的)，大家可以加一行代码试一下，我这里没有贴出来。
    这里直接用[]访问，vector 退化为数组，不会进行越界的判断。此时推荐使用 at()，会先进行越界检查。
    相关引申：
    针对 capacity 这个属性，STL 中的其他容器，如 list map set deque，由于这些容器的内存是散列分布的，因此不会发生类似 realloc() 的调用情况，因此我们可以认为 capacity 属性针对这些容器是没有意义的，因此设计时这些容器没有该属性。
    在 STL 中，拥有 capacity 属性的容器只有 vector 和 string。
    */
    return 0;
}