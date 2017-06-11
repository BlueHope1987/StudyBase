// ConsoleApplication.cpp : �������̨Ӧ�ó������ڵ㡣
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

//�Լ��Ĵ���
//����ͼ���� http://wenku.baidu.com/view/7f195105cc175527072208e3.html

//λͼ�ļ�ͷ���� 14�ֽ�
/*
typedef struct tagBITMAPFILEHEADER
{
WORD bfType; //λͼ�ļ����ͣ�������0x424D�����ַ�����BM����Ҳ����˵�����еġ�*.bmp���ļ���ͷ�����ֽڶ��ǡ�BM����
DWORD bfSize; //λͼ�ļ���С��������14���ֽڡ�
WORD bfReserved1; //Windows������
WORD bfReserved2; //Windows������
DWORD bfOffBits; //���ļ�ͷ��ʵ�ʵ�λͼ���ݵ�ƫ���ֽ���
} BITMAPFILEHEADER,FAR *LPBITMAPFILEHEADER, *PBITMAPFILEHEADER;
//λͼ��Ϣͷ 40�ֽ�
typedef struct tagBITMAPINFOHEADER
{
DWORD biSize; //���ṹ�ĳ��ȣ�Ϊ40���ֽڡ�
LONG biWidth; //λͼ�Ŀ�ȣ�������Ϊ��λ��
LONG biHeight; //λͼ�ĸ߶ȣ�������Ϊ��λ��
WORD biPlanes; //Ŀ���豸�ļ��𣬱�����1
WORD biBitCount; //ÿ��������ռ��λ����bit������ֵ����Ϊ1���ڰ�ͼ�񣩡�4��16ɫͼ����8��256ɫ����24�����ɫͼ�����µ�BMP��ʽ֧��32λɫ��
DWORD biCompression; //λͼѹ�����ͣ���Ч��ֵΪBI_RGB��δ��ѹ������BI_RLE8��BI_RLE4��BI_BITFILEDS����ΪWindows���峣����������ֻ����δ��ѹ�����������biCompression=BI_RGB��
DWORD biSizeImage; //ʵ�ʵ�λͼ����ռ�õ��ֽ�������ֵ�Ĵ�С�ڵ�4����λͼ�������о�����͡�
LONG biXPelsPerMeter; //ָ��Ŀ���豸��ˮƽ�ֱ��ʣ���λ������/�ס�
LONG biYPelsPerMeter; //ָ��Ŀ���豸�Ĵ�ֱ�ֱ��ʣ���λ������/�ס�
DWORD biClrUsed; //λͼʵ���õ�����ɫ���������ֵΪ�㣬���õ�����ɫ��Ϊ2��biBitCount���ݡ�
DWORD biClrImportant; //λͼ��ʾ��������Ҫ����ɫ���������ֵΪ�㣬����Ϊ���е���ɫ������Ҫ�ġ�
} BITMAPINFOHEADER, FAR *LPBITMAPINFOHEADER, *PBITMAPINFOHEADER;
//��ɫ�� ɫ��(4�ֽ�)���� ���ɫ����Ҫ��ɫ��
typedef struct tagRGBQUAD
{
BYTE rgbBlue; //��ɫ����
BYTE rgbGreen; //��ɫ����
BYTE rgbRed; //��ɫ����
BYTE rgbReserved; //�����ֽ�
}RGBQUAD;
*/

/*
��Ҫע�����㣺  ��һ��Windows�涨һ��ɨ������ռ���ֽ���������4�ı���������4�ı�����Ҫ����������䡣
����ͼ��Ŀ�ΪbiWidth�����ء�ÿ����biBitCount�����أ���һ��ɨ������ռ����ʵ�ֽ����ļ��㹫ʽ���£�
DataSizePerLine = (biWidth * biBitCount /8+ 3) / 4*4
��ô����ѹ�������λͼ���ݵĴ�С��BITMAPINFOHEADER�ṹ�е�biSizeImage��Ա���������£�
biSizeImage = DataSizePerLine * biHeight
�ڶ���һ����˵��BMP�ļ��������Ǵ�ͼ������½ǿ�ʼ����ɨ��ͼ��ģ������µ��ϡ������ң�
��ͼ�������ֵһһ��¼���������ͼ�����������ͼ�����½ǡ�

*/

//��BMP�߼�
unsigned char *pBmpBuf;//����ͼ�����ݵ�ָ��
int bmpWidth;//ͼ��Ŀ�
int bmpHeight;//ͼ��ĸ�
RGBQUAD *pColorTable;//��ɫ��ָ��
int biBitCount;//ͼ�����ͣ�ÿ����λ��
/************************
*�������ƣ�readBmp()
*������char *bmpName �ļ����ּ�·��
*����ֵ��0ʧ�� 1�ɹ�
*˵��������һ��ͼ���ļ�������·������ͼ���Ϊͽ���ݡ����ߡ���ɫ��ÿ����λ�������ݽ��ڴ棬�������Ӧ��ȫ�ֱ�����
************************/
bool readBmp(char *bmpName)
{
	//�����ƶ���ʽ���ƶ���ͼ���ļ�
	FILE *fp = fopen(bmpName, "rb");
	if (fp == 0) return 0;
	//����λͼ�ļ�ͷ�ṹ BITMAPFILEHEADER
	fseek(fp, sizeof(BITMAPFILEHEADER), 0);
	//����λͼ��Ϣͷ�ṹ��������ȡλͼ��Ϣͷ���ڴ棬����ڱ���head��
	BITMAPINFOHEADER head;
	fread(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	//��ȡͼ����ÿ������ռλ������Ϣ
	bmpWidth = head.biWidth;
	bmpHeight = head.biHeight;
	biBitCount = head.biBitCount;
	//�������������ͼ��ÿ��������ռ���ֽ�����������4�ı�����
	int lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;
	//�Ҷ�ͼ������ɫ������ɫ�����Ϊ256
	if (biBitCount == 8) {
		//������ɫ������Ҫ�Ŀռ䣬����ɫ����ڴ�
		pColorTable = new RGBQUAD[256];
		fread(pColorTable, sizeof(RGBQUAD), 256, fp);
	}
	//����λͼ��������Ҫ�Ŀռ䣬��λͼ���ݽ��ڴ�
	pBmpBuf = new unsigned char[lineByte * bmpHeight];
	fread(pBmpBuf, 1, lineByte * bmpHeight, fp);
	//�ر��ļ�
	fclose(fp);
	return 1;
}
/************************
*�������ƣ�saveBmp()
*������char *bmpName �ļ����ּ�·��
* unsigned char *imgBuf �����̵�λͼ����
* int width ������Ϊ��λ������λͼ�Ŀ�
* int height ������Ϊ��λ������λͼ��
* int biBitCount ÿ������ռλ��
* RGBQUAD *pColorTable ��ɫ��ָ��
*����ֵ��0ʧ�� 1�ɹ�
*˵��������һ��ͼ��λͼ���ݡ����ߡ���ɫ��ָ�뼰ÿ������ռ��λ������Ϣ������д��ָ���ļ���
************************/
bool saveBmp(char *bmpName, unsigned char *imgBuf, int width, int height, int biBitCount, RGBQUAD *pColorTable)
{
	//���λͼ����ָ��Ϊ0����û�����ݴ��룬��������
	if (!imgBuf)
		return 0;
	//��ɫ���С�����ֽ�Ϊ��λ���Ҷ�ͼ����ɫ��Ϊ1024�ֽڣ���ɫͼ����ɫ���СΪ0
	int colorTablesize = 0;
	if (biBitCount == 8)
		colorTablesize = 1024;
	//���洢ͼ������ÿ���ֽ���Ϊ4�ı���
	int lineByte = (width * biBitCount / 8 + 3) / 4 * 4;
	//�Զ�����д�ķ�ʽ���ļ�
	FILE *fp = fopen(bmpName, "wb");
	if (fp == 0) return 0;
	//����λͼ�ļ�ͷ�ṹ��������д�ļ�ͷ��Ϣ
	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42; //bmp����
							  //bfSize��ͼ���ļ�4����ɲ���֮��
	fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte*height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	//bfOffBits��ͼ���ļ�ǰ3����������ռ�֮��
	fileHead.bfOffBits = 54 + colorTablesize;
	//д�ļ�ͷ���ļ�
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
	//����λͼ��Ϣͷ�ṹ��������д��Ϣͷ��Ϣ
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
	//дλͼ��Ϣͷ���ڴ�
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	//����Ҷ�ͼ������ɫ��д���ļ�
	if (biBitCount == 8)
		fwrite(pColorTable, sizeof(RGBQUAD), 256, fp);
	//дλͼ���ݽ��ļ�
	fwrite(imgBuf, height*lineByte, 1, fp);
	//�ر��ļ�
	fclose(fp);
	return 1;
}

//�򵥵���
void main()
{

	//����ָ��BMP�ļ����ڴ�
	char readPath[] = "dog.BMP";
	readBmp(readPath);
	//���ͼ����Ϣ
	printf("width=%d,height=%d, biBitCount=%d\n", bmpWidth, bmpHeight, biBitCount);
	//ѭ��������ͼ�������
	int i, j;
	//ÿ���ֽ���
	int lineByte = (bmpWidth*biBitCount / 8 + 3) / 4 * 4;
	//ѭ����������Բ�ɫͼ�񣬱���ÿ���ص���������
	int k;
	//��ͼ�����½�1/4�����óɺ�ɫ
	if (biBitCount == 8) { //���ڻҶ�ͼ��
		for (i = 0; i < bmpHeight / 2; i++) {
			for (j = 0; j < bmpWidth / 2; j++) {
				*(pBmpBuf + i*lineByte + j) = 0;
			}
		}
	}
	else if (biBitCount == 24) { //��ɫͼ��
		for (i = 0; i < bmpHeight / 2; i++) {
			for (j = 0; j < bmpHeight / 2; j++) {
				*(pBmpBuf + i*lineByte + j * 3 + k) = 0;
			}
		}
	}
	//��ͼ�����ݴ���
	char writePath[] = "dogcpy.BMP";
	saveBmp(writePath, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	//�����������pBmpBuf��pColorTable��ȫ�ֱ��������ļ�����ʱ����Ŀռ�
	delete[] pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
	
	//��ɫBMP����
	//����ָ��BMP�ļ����ڴ�
	char readPath2[] = "dog_danse.BMP";
	readBmp(readPath2);
	//���ͼ����Ϣ
	printf("width=%d,height=%d, biBitCount=%d\n", bmpWidth, bmpHeight, biBitCount);
	//�ı�Ҷ�ͼ�����ɫ����ɫ������ֵ���鿴ǰ��仯
	if(biBitCount==8){
		for (i = 0; i < 256; i++) {
			pColorTable[i].rgbBlue = 255 - pColorTable[i].rgbBlue;
		}
	}
	//��ͼ�����ݴ���
	char writePath2[] = "dogcpy_danse.BMP";
	saveBmp(writePath2, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	//�����������pBmpBuf��pColorTable��ȫ�ֱ��������ļ�����ʱ����Ŀռ�
	delete[]pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
}
//P18
//�Լ��Ĵ������
