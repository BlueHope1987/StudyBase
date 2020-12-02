// 斐波那契数列.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>

int fib(int i) {
	if (i < 2)
		return i == 0 ? 0 : 1;
	return fib(i - 2) + fib(i - 1);
}
int main()
{
	int i = 0, f = 0;
	for (i = 0; i < 40; i++) {
		f = fib(i);
		printf("%d", f);
	}
    return 0;
}

