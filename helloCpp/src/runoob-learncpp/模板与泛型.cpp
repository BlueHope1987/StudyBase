/*
https://www.runoob.com/cplusplus/cpp-templates.html
模板是泛型编程的基础，泛型编程即以一种独立于任何特定类型的方式编写代码。
模板是创建泛型类或函数的蓝图或公式。库容器，比如迭代器和算法，都是泛型编程的例子，它们都使用了模板的概念。
每个容器都有一个单一的定义，比如 向量，我们可以定义许多不同类型的向量，比如 vector <int> 或 vector <string>。
您可以使用模板来定义函数和类，接下来让我们一起来看看如何使用。
*/
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
using namespace std;

//inline 内联函数 简单函数体 提升执行效率 不能包含复杂结构控制 对编译器建议 看编译器意思将函数内联到调用程序内以减少栈占用
//定义在类中的成员函数缺省都是内联的
//https://www.cnblogs.com/fnlingnzb-learner/p/6423917.html
//const常量 &引用
//用引用可以减少数值传递过程中的时间，而const则保证这个传过来的值在使用时不被改变。
//参数声明中const，const string parameter表明复制出来的副本（也就是形参）你不会修改。const string & parameter表明你不会对调用函数的实参进行修改
//https://blog.csdn.net/xiongchengluo1129/article/details/79123487

//函数模板
template <typename T>
inline T const &Max(T const &a, T const &b)
{
    return a < b ? b : a;
    //表达式?真:假
}

//类模板
template <class T>
class Stack
{
private:
    vector<T> elems; // 元素

public:
    void push(T const &); // 入栈
    void pop();           // 出栈
    T top() const;        // 返回栈顶元素
    bool empty() const
    { // 如果为空则返回真。
        return elems.empty();
    }
};

template <class T>
void Stack<T>::push(T const &elem)
{
    // 追加传入元素的副本
    elems.push_back(elem);
}

template <class T>
void Stack<T>::pop()
{
    if (elems.empty())
    {
        throw out_of_range("Stack<>::pop(): empty stack");
    }
    // 删除最后一个元素
    elems.pop_back();
}

template <class T>
T Stack<T>::top() const
{
    if (elems.empty())
    {
        throw out_of_range("Stack<>::top(): empty stack");
    }
    // 返回最后一个元素的副本
    return elems.back();
}

int main()
{

    //函数模板测试
    int i = 39;
    int j = 20;
    cout << "Max(i, j): " << Max(i, j) << endl;

    double f1 = 13.5;
    double f2 = 20.7;
    cout << "Max(f1, f2): " << Max(f1, f2) << endl;

    string s1 = "Hello";
    string s2 = "World";
    cout << "Max(s1, s2): " << Max(s1, s2) << endl;

    //类模板测试
    try
    {
        Stack<int> intStack;       // int 类型的栈
        Stack<string> stringStack; // string 类型的栈

        // 操作 int 类型的栈
        intStack.push(7);
        cout << intStack.top() << endl;

        // 操作 string 类型的栈
        stringStack.push("hello");
        cout << stringStack.top() << std::endl;
        stringStack.pop();
        stringStack.pop();
    }
    catch (exception const &ex)
    {
        cerr << "Exception: " << ex.what() << endl;
        return -1;
        cin>>i;
    }
    return 0;
}