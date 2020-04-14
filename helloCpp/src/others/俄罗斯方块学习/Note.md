方块枚举
I	[0,0,0,0]	[0,1,0,0]
    [1,1,1,1]	[0,1,0,0]   [0,1,0,0]
    [0,0,0,0]	[0,1,0,0]
    [0,0,0,0]	[0,1,0,0]

O	[1,1]
    [1,1]

S	[0,1,1]	[1,0,0]
    [1,1,0]	[1,1,0]     [0,1,0]
    [0,0,0] [0,1,0]

Z	[1,1,0]	[0,1,0]
    [0,1,1]	[1,1,0]
    [0,0,0] [1,0,0]

L	[1,0,0]	[0,1,1]	[0,0,0]	[0,1,0]
    [1,1,1]	[0,1,0]	[1,1,1]	[0,1,0]   [0,1,0]
    [0,0,0] [0,1,0]	[0,0,1]	[1,1,0]

J	[0,0,1]	[1,0]	[1,1,1]	[1,1]
    [1,1,1]	[1,0]	[1,0,0]	[0,1]
            [1,1]			[0,1]

T	[0,1,0] [0,1,0]	[0,0,0]	[0,1,0]
    [1,1,1]	[0,1,1]	[1,1,1]	[1,1,0]   [0,1,0]
    [0,0,0] [0,1,0]	[0,1,0]	[0,1,0]


// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

class tetrisView {
protected:
public:
	unsigned char** tetrisMap; //战场二维数组
	unsigned char*	tetrisHsyLst; //历史记录
	short width = 10; //战场宽
	short height = 20; //战场高
	unsigned char curtetris, curangle,nxttetris; //当前方块 当前角度 下个方块
	short curx, cury;//当前坐标
	unsigned char speed;//速度 多少帧一跳

	tetrisView(short x,short y,int qsize) {
		//初始化 重置 xy战场大小 qsize队列大小
		if (x>5 & x<255) width=x;
		if (y>10 & y<255) height = y;
		if (qsize > 50 & qsize < 1000)
			tetrisHsyLst = new unsigned char[qsize];
		else
			tetrisHsyLst = new unsigned char[200];
		tetrisMap = new unsigned char*[width];
		for (short i = 0; i < width; i++){
			tetrisMap[i] = new unsigned char[height];
			for (short j = 0; j < height; j++)
				tetrisMap[i][j] = 0;
		}
	}

	void pullNextTetris(unsigned char nxtblk) {
		//推入下个方块 IOSZLJT
		nxttetris = nxtblk;
	}

	void updateframe(int operate) {
		//更新帧 operate操作 上下左右空
	}

	~tetrisView() {

	}
};

int main()
{
	tetrisView game = tetrisView(0, 0, 0);
    return 0;
}

