// 斐波那契数列.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>

void print() {
	char c;
	scanf("%c", &c);
	if ('#' != c) {
		print();
	}
	if ('#' != c) {
		printf("%c", c);
	}
}
int main()
{
	print();
	return 0;
}

