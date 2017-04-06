// 斐波那契数列.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>

int main()
{
	int a[40], i = 0, s = 0, n = 30;
	a[0] = 0;
	a[1] = 1;
	for (i = 2; i < 40; i++) {
		a[i] = a[i - 2] + a[i - 1];
	}
	for (i = 0; i < 40; i++) {
		printf("%d", a[i]);
	}
    return 0;
}

