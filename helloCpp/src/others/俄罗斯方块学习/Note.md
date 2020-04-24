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

#include <iostream>

using namespace std;

enum tBlock_set {I,O,S,Z,L,J,T};
enum operate_set {up,down,left,right,fall};

class tetrisView {
private:
	bool Blocks[7][2][4] = {
							{{0,0,0,0},{1,1,1,1}},
							{{0,1,1,0},{0,1,1,0}},
							{{0,1,1,0},{1,1,0,0}},
							{{1,1,0,0},{0,1,1,0}},
							{{1,0,0,0},{1,1,1,0}},
							{{0,0,1,0},{1,1,1,0}},
							{{0,1,0,0},{1,1,1,0}}
	};
	short blockofset[7][4][2]={
                            {{1,1},{1,1},{-1,-1},{-1,-1}},
                            {{0,1},{-1,-1},{-1,-1},{-1,-1}},
                            {{0,1},{0,0},{-1,-1},{-1,-1}},
                            {{0,1},{0,0},{-1,-1},{-1,-1}},
                            {{0,1},{1,1},{0,1},{1,1}},
                            {{0,1},{1,1},{0,1},{1,1}},
                            {{1,1},{1,1},{1,1},{1,1}},
	};//每方块每角度原点YX偏移+ -1为角度不可用

protected:
public:
	unsigned char** tetrisMap; //战场二维数组
	unsigned char*	tetrisHsyLst; //历史记录
	short width = 10; //战场宽
	short height = 20; //战场高
	unsigned char curangle,lstangle; //当前角度 上个角度
	tBlock_set curtetris, nxttetris; //当前方块 当前角度 下个方块
	short curx, cury,lstx,lsty;//当前坐标 上个坐标
	unsigned char speed;//速度 多少帧一跳

	tetrisView(short x, short y, int qsize) {
		//初始化 重置 xy战场大小 qsize队列大小
		if (x>5 && x<255) width = x;
		if (y>10 && y<255) height = y;
		if (qsize > 50 & qsize < 1000)
			tetrisHsyLst = new unsigned char[qsize];
		else
			tetrisHsyLst = new unsigned char[200];
		tetrisMap = new unsigned char*[width];
		for (short i = 0; i < height; i++) {
			tetrisMap[i] = new unsigned char[height];
			for (short j = 0; j < width; j++)
				tetrisMap[i][j] = 0;
		}

		//test
		nxttetris=I;
		cury = -1;
		updateframe(up);
	}

	void pullNextTetris(tBlock_set nxtblk) {
		//推入下个方块 IOSZLJT
		nxttetris = nxtblk;
	}

	void updateframe(operate_set operate) {
		//更新帧 operate操作 上下左右空
		//当cury=-1新方块 重置变量
		if (cury == -1) {
			curtetris = nxttetris;
			cury = height;
			curx = width / 2 - 1;
			curangle=0;//复位方块角度
			lstx=-1;
			lsty=-1;
			lstangle=0;
		}
		//角度为0时的逻辑 1、验证当前角度、坐标，2、消除上个角度、坐标 或 固定
		if(curangle==0){
            //检查
            for(short i=0;i<2;i++)
            {
                for(short j=0;j<4;j++)
                {
                        //如果方块当前格和战场当前格均为真(碰撞) 则固定上一个（重置）
                    if(
                       Blocks[curtetris][i][j]&&
                       tetrisMap[cury-blockofset[curtetris][curangle][0]+i]
                                [curx-blockofset[curtetris][curangle][0]+j]
                    ){
                        cury == -1;
                        return;
                     };
                };

            };
            //消除
            for(short i=0;i<2;i++)
            {
                for(short j=0;j<4;j++)
                {
                };
            };
			//绘新
		}




	}

	~tetrisView() {

	}
};

int main()
{
	tetrisView game = tetrisView(0, 0, 0);
	return 0;
}
