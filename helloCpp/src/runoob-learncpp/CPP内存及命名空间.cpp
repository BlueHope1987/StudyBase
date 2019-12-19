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

class Box
{
public:
    Box()
    {
        cout << "调用构造函数！" << endl;
    }
    ~Box()
    {
        cout << "调用析构函数！" << endl;
    }
};

namespace first_space
{
void func()
{
    cout << "Inside first_space" << endl;
}
// 第二个命名空间
namespace second_space
{
void func()
{
    cout << "Inside second_space" << endl;
}
} // namespace second_space
} // namespace first_space
using namespace first_space::second_space;

int main()
{

    double *pvalue = NULL;
    if (!(pvalue = new double))
    {
        cout << "Error: out of memory." << endl;
        exit(1);
    };
    delete pvalue;

    //三维数组
    int i, j, k; // p[2][3][4]

    int ***p;
    p = new int **[2];
    for (i = 0; i < 2; i++)
    {
        p[i] = new int *[3];
        for (j = 0; j < 3; j++)
            p[i][j] = new int[4];
    }

    //输出 p[i][j][k] 三维数据
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 4; k++)
            {
                p[i][j][k] = i + j + k;
                cout << p[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // 释放内存
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            delete[] p[i][j];
        }
    }
    for (i = 0; i < 2; i++)
    {
        delete[] p[i];
    }
    delete[] p;

    Box *myBoxArray = new Box[4];
    delete[] myBoxArray; // 删除数组

    // 调用第二个命名空间中的函数
    func();

    cin >> i;

    return 0;
}