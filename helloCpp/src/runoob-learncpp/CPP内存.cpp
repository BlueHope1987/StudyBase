#include <iostream>
#include <vector>
using namespace std;

//https://www.runoob.com/cplusplus/cpp-dynamic-memory.html
//C++动态内存
/*
了解动态内存在 C++ 中是如何工作的是成为一名合格的 C++ 程序员必不可少的。C++ 程序中的内存分为两个部分：
栈：在函数内部声明的所有变量都将占用栈内存。
堆：这是程序中未使用的内存，在程序运行时可用于动态分配内存。
很多时候，您无法提前预知需要多少内存来存储某个定义变量中的特定信息，所需内存的大小需要在运行时才能确定。
在 C++ 中，您可以使用特殊的运算符为给定类型的变量在运行时分配堆内的内存，这会返回所分配的空间地址。这种运算符即 new 运算符。
如果您不再需要动态分配的内存空间，可以使用 delete 运算符，删除之前由 new 运算符分配的内存。
*/

int main()
{

    double *pvalue = NULL;
    if (!(pvalue = new double))
    {
        cout << "Error: out of memory." << endl;
        exit(1);
    };
    delete pvalue;


    //二维数组
    int **array;
    // 假定数组第一维长度为 m， 第二维长度为 n
    // 动态分配空间
    array = new int *[m];
    for (int i = 0; i < m; i++)
    {
        array[i] = new int[n];
    }
    //释放
    for (int i = 0; i < m; i++)
    {
        delete[] arrary[i];
    }
    delete[] array;

    return 0;
}