// 쳲���������.cpp : �������̨Ӧ�ó������ڵ㡣
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

