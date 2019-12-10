#include <iostream>
#include <stdio.h>
#include <windows.h>

int color(int num) //num为每一种颜色所代表的数字，范围时0~15
{
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), num);
    return 0;
}
void gotoxy(int x, int y)
{
    COORD Pos;
    Pos.X = x; //横坐标
    Pos.Y = y; //纵坐标
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), Pos);
}

int main()
{
    std::cout << "Hello Easy C++ project!" << std::endl;
    printf("hello console");

    int i;
    for (i = 0; i <= 15; i++)
    {
        color(i);
        printf("这是第%d号颜色\n", i);
    }

    for (i = 0; i < 15; i++)
    {
        gotoxy(50, i);
        printf("hello world!");
    }

    system("pause");
    return 0;
}
