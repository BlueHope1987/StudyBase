// 斐波那契数列.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#define size 10
typedef int etype;

int fib(int i) {
	if (i < 2)
		return i == 0 ? 0 : 1;
	return fib(i - 2) + fib(i - 1);
}
int cz(int n, int *p, int low, int high) {
	etype mid;
	mid = low + (high - low) / 2;
	if (*(p + mid) > n) {
		return cz(n, p, low, mid);
	}
	if (*(p + mid) < n) {
		return cz(n, p, mid, high);
	}
	else {
		return mid;
	}
}
int main()
{
	etype n, a[size], i;
	for (i = 0; i < size; i++) {
		a[i] = fib(i);
	}
	printf("请输入查找的数：");
	scanf("%d", &n);
	i = cz(n, a, 0, size);
	printf("你要查找的数在斐波那契数列中第%d个\n", i + 1);
	return 0;
}

