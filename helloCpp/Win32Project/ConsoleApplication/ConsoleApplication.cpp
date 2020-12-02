// ConsoleApplication.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <stdio.h>
#include <Windows.h>
/*
#include "iostream"
using namespace std;

int main()
{
	cout << "Hello world" << endl;
	int a;
	cin >> a;
    return 0;
}
*/

//自己的代码
//数字图像处理 http://wenku.baidu.com/view/7f195105cc175527072208e3.html

//位图文件头定义 14字节
/*
typedef struct tagBITMAPFILEHEADER
{
WORD bfType; //位图文件类型，必须是0x424D，即字符串“BM”，也就是说，所有的“*.bmp”文件的头两个字节都是“BM”。
DWORD bfSize; //位图文件大小，包括这14个字节。
WORD bfReserved1; //Windows保留字
WORD bfReserved2; //Windows保留字
DWORD bfOffBits; //从文件头到实际的位图数据的偏移字节数
} BITMAPFILEHEADER,FAR *LPBITMAPFILEHEADER, *PBITMAPFILEHEADER;
//位图信息头 40字节
typedef struct tagBITMAPINFOHEADER
{
DWORD biSize; //本结构的长度，为40个字节。
LONG biWidth; //位图的宽度，以像素为单位。
LONG biHeight; //位图的高度，以像素为单位。
WORD biPlanes; //目标设备的级别，必须是1
WORD biBitCount; //每个像素所占的位数（bit），其值必须为1（黑白图像）、4（16色图）、8（256色）、24（真彩色图），新的BMP格式支持32位色。
DWORD biCompression; //位图压缩类型，有效的值为BI_RGB（未经压缩）、BI_RLE8、BI_RLE4、BI_BITFILEDS（均为Windows定义常量）。这里只讨论未经压缩的情况，即biCompression=BI_RGB。
DWORD biSizeImage; //实际的位图数据占用的字节数，该值的大小在第4部分位图数据中有具体解释。
LONG biXPelsPerMeter; //指定目标设备的水平分辨率，单位是像素/米。
LONG biYPelsPerMeter; //指定目标设备的垂直分辨率，单位是像素/米。
DWORD biClrUsed; //位图实际用到的颜色数，如果该值为零，则用到的颜色数为2的biBitCount次幂。
DWORD biClrImportant; //位图显示过程中重要的颜色数，如果该值为零，则认为所有的颜色都是重要的。
} BITMAPINFOHEADER, FAR *LPBITMAPINFOHEADER, *PBITMAPINFOHEADER;
//颜色表 色点(4字节)数组 真彩色不需要颜色表
typedef struct tagRGBQUAD
{
BYTE rgbBlue; //蓝色分量
BYTE rgbGreen; //绿色分量
BYTE rgbRed; //红色分量
BYTE rgbReserved; //保留字节
}RGBQUAD;
*/

/*
需要注意两点：  第一，Windows规定一个扫描行所占的字节数必须是4的倍数，不足4的倍数则要对其进行扩充。
假设图像的宽为biWidth个像素、每像素biBitCount个比特，其一个扫描行所占的真实字节数的计算公式如下：
DataSizePerLine = (biWidth * biBitCount /8+ 3) / 4*4
那么，不压缩情况下位图数据的大小（BITMAPINFOHEADER结构中的biSizeImage成员）计算如下：
biSizeImage = DataSizePerLine * biHeight
第二，一般来说，BMP文件的数据是从图像的左下角开始逐行扫描图像的，即从下到上、从左到右，
将图像的像素值一一记录下来，因此图像坐标零点在图像左下角。

*/

//读BMP逻辑
unsigned char *pBmpBuf;//读入图像数据的指针
int bmpWidth;//图像的宽
int bmpHeight;//图象的高
RGBQUAD *pColorTable;//颜色表指针
int biBitCount;//图像类型，每像素位数
/************************
*函数名称：readBmp()
*参数：char *bmpName 文件名字及路径
*返回值：0失败 1成功
*说明：给定一个图像文件名及其路径，读图像的为徒数据、宽、高、颜色表及每像素位数等数据进内存，存放在相应的全局变量中
************************/
bool readBmp(char *bmpName)
{
	//二进制都方式打开制定的图像文件
	FILE *fp = fopen(bmpName, "rb");
	if (fp == 0) return 0;
	//跳过位图文件头结构 BITMAPFILEHEADER
	fseek(fp, sizeof(BITMAPFILEHEADER), 0);
	//定义位图信息头结构变量，读取位图信息头进内存，存放在变量head中
	BITMAPINFOHEADER head;
	fread(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	//获取图像宽高每像素所占位数等信息
	bmpWidth = head.biWidth;
	bmpHeight = head.biHeight;
	biBitCount = head.biBitCount;
	//定义变量，计算图像每行像素所占的字节数（必须是4的倍数）
	int lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;
	//灰度图像有颜色表，且颜色表表项为256
	if (biBitCount == 8) {
		//申请颜色表所需要的空间，读颜色表进内存
		pColorTable = new RGBQUAD[256];
		fread(pColorTable, sizeof(RGBQUAD), 256, fp);
	}
	//申请位图数据所需要的空间，读位图数据进内存
	pBmpBuf = new unsigned char[lineByte * bmpHeight];
	fread(pBmpBuf, 1, lineByte * bmpHeight, fp);
	//关闭文件
	fclose(fp);
	return 1;
}
/************************
*函数名称：saveBmp()
*参数：char *bmpName 文件名字及路径
* unsigned char *imgBuf 待存盘的位图数据
* int width 以像素为单位待存盘位图的宽
* int height 以像素为单位待存盘位图高
* int biBitCount 每像素所占位数
* RGBQUAD *pColorTable 颜色表指针
*返回值：0失败 1成功
*说明：给定一个图像位图数据、宽、高、颜色表指针及每像素所占的位数等信息，将其写到指定文件中
************************/
bool saveBmp(char *bmpName, unsigned char *imgBuf, int width, int height, int biBitCount, RGBQUAD *pColorTable)
{
	//如果位图数据指针为0，则没有数据传入，函数返回
	if (!imgBuf)
		return 0;
	//颜色表大小，以字节为单位，灰度图像颜色表为1024字节，彩色图像颜色表大小为0
	int colorTablesize = 0;
	if (biBitCount == 8)
		colorTablesize = 1024;
	//待存储图像数据每行字节数为4的倍数
	int lineByte = (width * biBitCount / 8 + 3) / 4 * 4;
	//以二进制写的方式打开文件
	FILE *fp = fopen(bmpName, "wb");
	if (fp == 0) return 0;
	//申请位图文件头结构变量，填写文件头信息
	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42; //bmp类型
							  //bfSize是图像文件4个组成部分之和
	fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte*height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	//bfOffBits是图像文件前3个部分所需空间之和
	fileHead.bfOffBits = 54 + colorTablesize;
	//写文件头进文件
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
	//申请位图信息头结构变量，填写信息头信息
	BITMAPINFOHEADER head;
	head.biBitCount = biBitCount;
	head.biClrImportant = 0;
	head.biClrUsed = 0;
	head.biCompression = 0;
	head.biHeight = height;
	head.biPlanes = 1;
	head.biSize = 40;
	head.biSizeImage = lineByte*height;
	head.biWidth = width;
	head.biXPelsPerMeter = 0;
	head.biYPelsPerMeter = 0;
	//写位图信息头进内存
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	//如果灰度图像，有颜色表，写入文件
	if (biBitCount == 8)
		fwrite(pColorTable, sizeof(RGBQUAD), 256, fp);
	//写位图数据进文件
	fwrite(imgBuf, height*lineByte, 1, fp);
	//关闭文件
	fclose(fp);
	return 1;
}

//简单调用
void main()
{

	//读入指定BMP文件进内存
	char readPath[] = "dog.BMP";
	readBmp(readPath);
	//输出图像信息
	printf("width=%d,height=%d, biBitCount=%d\n", bmpWidth, bmpHeight, biBitCount);
	//循环变量，图像的坐标
	int i, j;
	//每行字节数
	int lineByte = (bmpWidth*biBitCount / 8 + 3) / 4 * 4;
	//循环变量，针对彩色图像，遍历每像素的三个分量
	int k;
	//将图像左下角1/4部分置成黑色
	if (biBitCount == 8) { //对于灰度图像
		for (i = 0; i < bmpHeight / 2; i++) {
			for (j = 0; j < bmpWidth / 2; j++) {
				*(pBmpBuf + i*lineByte + j) = 0;
			}
		}
	}
	else if (biBitCount == 24) { //彩色图像
		for (i = 0; i < bmpHeight / 2; i++) {
			for (j = 0; j < bmpHeight / 2; j++) {
				*(pBmpBuf + i*lineByte + j * 3 + k) = 0;
			}
		}
	}
	//将图像数据存盘
	char writePath[] = "dogcpy.BMP";
	saveBmp(writePath, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	//清除缓冲区，pBmpBuf和pColorTable是全局变量，在文件读入时申请的空间
	delete[] pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
	
	//单色BMP处理
	//读入指定BMP文件进内存
	char readPath2[] = "dog_danse.BMP";
	readBmp(readPath2);
	//输出图像信息
	printf("width=%d,height=%d, biBitCount=%d\n", bmpWidth, bmpHeight, biBitCount);
	//改变灰度图像的颜色表蓝色分量的值，查看前后变化
	if(biBitCount==8){
		for (i = 0; i < 256; i++) {
			pColorTable[i].rgbBlue = 255 - pColorTable[i].rgbBlue;
		}
	}
	//将图像数据存盘
	char writePath2[] = "dogcpy_danse.BMP";
	saveBmp(writePath2, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	//清除缓冲区，pBmpBuf和pColorTable是全局变量，在文件读入时申请的空间
	delete[]pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
}
//P18
//自己的代码完毕
