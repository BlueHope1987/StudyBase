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

enum tBlock_set { I, O, S, Z, L, J, T };
enum operate_set { op_up, op_down, op_left, op_right, op_fall, null };

class tetrisView
{
private:
	bool Blocks[7][2][4] =
	{
		{ { 0,0,0,0 },{ 1,1,1,1 } },
		{ { 0,1,1,0 },{ 0,1,1,0 } },
		{ { 0,1,1,0 },{ 1,1,0,0 } },
		{ { 1,1,0,0 },{ 0,1,1,0 } },
		{ { 1,0,0,0 },{ 1,1,1,0 } },
		{ { 0,0,1,0 },{ 1,1,1,0 } },
		{ { 0,1,0,0 },{ 1,1,1,0 } }
	};//Blocks[编号][Y][X]
	short blockofset[7][4][2] =
	{
		{ { 1,1 },{ 1,1 },{ -1,-1 },{ -1,-1 } },
		{ { 0,1 },{ -1,-1 },{ -1,-1 },{ -1,-1 } },
		{ { 0,1 },{ 0,0 },{ -1,-1 },{ -1,-1 } },
		{ { 0,1 },{ 0,0 },{ -1,-1 },{ -1,-1 } },
		{ { 0,1 },{ 1,1 },{ 0,1 },{ 1,1 } },
		{ { 0,1 },{ 1,1 },{ 0,1 },{ 1,1 } },
		{ { 1,1 },{ 1,1 },{ 1,1 },{ 1,1 } },
	};//blockofset[编号][角度][YX偏移] 每方块每角度原点YX偏移+ -1为角度不可用

	//当前和先前方块每格坐标，缓存消除用，避免重复计算
	short curblkloc[8][2], lstblkloc[8][2];
protected:
public:
	unsigned char** tetrisMap; //战场二维数组
	unsigned char*	tetrisHsyLst; //历史记录
	short width = 10; //战场宽
	short height = 20; //战场高
	unsigned char curangle, lstangle; //当前角度 上个角度
	tBlock_set curtetris, nxttetris; //当前方块 当前角度 下个方块
	short curx, cury, lstx, lsty;//当前坐标 上个坐标
	unsigned char speed;//速度 多少帧一跳

	tetrisView(short x, short y, int qsize)
	{
		//初始化 重置 xy战场大小 qsize队列大小
		if (x > 5 && x < 255)
			width = x;
		if (y > 10 && y < 255)
			height = y;
		if (qsize > 50 & qsize < 1000)
			tetrisHsyLst = new unsigned char[qsize];
		else
			tetrisHsyLst = new unsigned char[200];
		tetrisMap = new unsigned char*[width];
		for (short i = 0; i < height; i++)
		{
			tetrisMap[i] = new unsigned char[height];
			for (short j = 0; j < width; j++)
				tetrisMap[i][j] = 0;
		}

		//test
		nxttetris = I;
		cury = -1;
		updateframe(op_up);
	}

	void pullNextTetris(tBlock_set nxtblk)
	{
		//推入下个方块 IOSZLJT
		nxttetris = nxtblk;
	}

	void updateframe(operate_set operate)
	{
		//更新帧 operate操作 上下左右空
		//处理操作、将操作定为偏移
		short _opx, _opy;
		switch (operate)
		{
		case op_up:
			//复位操作偏移
			_opx = 0;
			_opy = 0;
			//角度0-3轮换
			if (curangle < 3)
			{
				curangle++;
			}
			else
			{
				curangle = 0;
			};
			break;
		case op_down:
			_opx = 0;
			_opy = -1;
			break;
		case op_left:
			_opx = -1;
			_opy = 0;
			break;
		case op_right:
			_opx = 1;
			_opy = 0;
			break;
		case op_fall:
			//TODO:直接落下逻辑
			break;
		default:
			_opx = 0;
			_opy = 0;
			break;
		}
		//当cury=-1新方块 重置变量
		if (cury == -1)
		{
			curtetris = nxttetris;
			cury = height - 1;//数组值需-1
			curx = width / 2 - 1;
			curangle = 0;//复位方块角度
			lstx = -1;
			lsty = -1;
			lstangle = 0;
			lstblkloc[0][0] = -1;
		}
		short bkn = 0;//curblkloc角标

		//角度为0时的逻辑 1、验证当前角度、坐标，2、消除上个角度、坐标 或 固定
		switch (curangle)
		{
		case 0:
			//检查
			for (short i = 0; i < 2; i++)
			{
				for (short j = 0; j < 4; j++)
				{

					short _cy = cury + blockofset[curtetris][curangle][0] - i + _opy;
					short _cx = curx - blockofset[curtetris][curangle][1] + j + _opx;

					//如果方块当前格为真
					if (Blocks[curtetris][i][j])
					{
						//撞底
						if (_cy < 0)
						{
							cury = -1;
							return;
						}
						//超顶格矫正Y
						while (_cy > height - 1)
						{
							cury--;
							_cy--;
						}
						//未撞墙检查
						if (_cx <= width - 1 && _cx >= 0)
						{
							//如果战场当前格为真(碰撞) 则固定上一个（重置）
							if (tetrisMap[_cy][_cx])
							{
								cury = -1;
								return;
							}
						}
						else
						{
							//如果撞墙 返回无效操作
							return;
						}
						curblkloc[bkn][0] = _cy;
						curblkloc[bkn][1] = _cx;
						bkn++;
					};
				};
			};
			if (bkn < 8) curblkloc[bkn][0] = -1;//结尾标记
			break;

		case 1:
			//检查
			//如果角度不可用 返回继续切换角度
			if (blockofset[curtetris][curangle][0] == -1) {
				return updateframe(op_up);
			}
			for (short i = 0; i < 2; i++)
			{
				for (short j = 0; j < 4; j++)
				{

					short _cy = cury + blockofset[curtetris][curangle][0] - j + _opy;
					short _cx = curx - blockofset[curtetris][curangle][1] + i + _opx;

					//如果方块当前格为真
					if (Blocks[curtetris][1-i][j])
					{
						//撞底
						if (_cy < 0)
						{
							cury = -1;
							return;
						}
						//超顶格矫正Y
						while (_cy > height - 1)
						{
							cury--;
							_cy--;
						}
						//未撞墙检查
						if (_cx <= width - 1 && _cx >= 0)
						{
							//如果战场当前格为真(碰撞) 则固定上一个（重置）
							if (tetrisMap[_cy][_cx])
							{
								cury = -1;
								return;
							}
						}
						else
						{
							//如果撞墙 返回无效操作
							return;
						}
						curblkloc[bkn][0] = _cy;
						curblkloc[bkn][1] = _cx;
						bkn++;
					};
				};
			};
			if (bkn < 8) curblkloc[bkn][0] = -1;//结尾标记
			break;

		case 2:
			//检查
			//如果角度不可用 返回继续切换角度
			if (blockofset[curtetris][curangle][0] == -1) {
				return updateframe(op_up);
			}
			for (short i = 0; i < 2; i++)
			{
				for (short j = 0; j < 4; j++)
				{

					short _cy = cury + blockofset[curtetris][curangle][0] - i + _opy;
					short _cx = curx - blockofset[curtetris][curangle][1] + j + _opx;

					//如果方块当前格为真
					if (Blocks[curtetris][1-i][3-j])
					{
						//撞底
						if (_cy < 0)
						{
							cury = -1;
							return;
						}
						//超顶格矫正Y
						while (_cy > height - 1)
						{
							cury--;
							_cy--;
						}
						//未撞墙检查
						if (_cx <= width - 1 && _cx >= 0)
						{
							//如果战场当前格为真(碰撞) 则固定上一个（重置）
							if (tetrisMap[_cy][_cx])
							{
								cury = -1;
								return;
							}
						}
						else
						{
							//如果撞墙 返回无效操作
							return;
						}
						curblkloc[bkn][0] = _cy;
						curblkloc[bkn][1] = _cx;
						bkn++;
					};
				};
			};
			if (bkn < 8) curblkloc[bkn][0] = -1;//结尾标记
			break;

		case 3:
			//检查
			//如果角度不可用 返回继续切换角度
			if (blockofset[curtetris][curangle][0] == -1) {
				return updateframe(op_up);
			}
			for (short i = 0; i < 2; i++)
			{
				for (short j = 0; j < 4; j++)
				{

					short _cy = cury + blockofset[curtetris][curangle][0] - j + _opy;
					short _cx = curx - blockofset[curtetris][curangle][1] + i + _opx;

					//如果方块当前格为真
					if (Blocks[curtetris][i][3 - j])
					{
						//撞底
						if (_cy < 0)
						{
							cury = -1;
							return;
						}
						//超顶格矫正Y
						while (_cy > height - 1)
						{
							cury--;
							_cy--;
						}
						//未撞墙检查
						if (_cx <= width - 1 && _cx >= 0)
						{
							//如果战场当前格为真(碰撞) 则固定上一个（重置）
							if (tetrisMap[_cy][_cx])
							{
								cury = -1;
								return;
							}
						}
						else
						{
							//如果撞墙 返回无效操作
							return;
						}
						curblkloc[bkn][0] = _cy;
						curblkloc[bkn][1] = _cx;
						bkn++;
					};
				};
			};
			if (bkn < 8) curblkloc[bkn][0] = -1;//结尾标记
			break;

		default:
			break;
		}


		//消除
		for (short i = 0; i < 8 && lstblkloc[i][0] != -1; i++)
		{
			if (tetrisMap[lstblkloc[i][0]][lstblkloc[i][1]] == 1) {
				tetrisMap[lstblkloc[i][0]][lstblkloc[i][1]] = 0;
			}
			else {
				throw;
			}
		};
		//绘新
		for (short i = 0; i < 8 && curblkloc[i][0] != -1; i++)
		{
			if (tetrisMap[curblkloc[i][0]][curblkloc[i][1]] == 0) {
				tetrisMap[curblkloc[i][0]][curblkloc[i][1]] = 1;
				lstblkloc[i][0] = curblkloc[i][0];
				curblkloc[i][0] = -1;
				lstblkloc[i][1] = curblkloc[i][1];
			}
			else {
				throw;
			}
		};




	}

	~tetrisView()
	{

	}
};

int main()
{
	tetrisView game = tetrisView(0, 0, 0);
	char ky;
	for (int i = 1; i == 1;) {
		cin >> ky;
		switch (ky) {
		case 'w':
			game.updateframe(op_up);
			break;
		case 'a':
			game.updateframe(op_left);
			break;
		case 'd':
			game.updateframe(op_right);
			break;
		case 's':
			game.updateframe(op_down);
			break;
		case 'f':
			game.updateframe(op_fall);
			break;
		case 'q':
			return 0;
		default:
			break;
		}
	}
	return 0;
}