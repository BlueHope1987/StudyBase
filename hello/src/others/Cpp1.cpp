//��Ȩ����(C)ε��ϣ��������
//��Ѹ�����ݽṹ��ҵ

#include "stdio.h"
#include "iostream.h"

typedef struct BiTNode{
	char data;
	struct BiTNode *lchild,*rchild;
}BiTNode,*BiTree;

char Initiate(BiTree bt){
	bt=new BiTNode;
	if(!bt)
	return 0;
	bt->lchild=NULL;
	bt->rchild=NULL;
	return 1;
}

BiTree InsertL(BiTree bt,char x,BiTree parent){
	BiTree p;
	if(parent==NULL){
		cout<<"�������"<<endl;
		return NULL;
	}
	p=new BiTNode;
	if(!p)
		return NULL;
	p->data=x;
	p->lchild=NULL;
	p->rchild=NULL;
	if(parent->lchild==NULL)
		parent->lchild=p;
	else{
			p->lchild=parent->lchild;
			parent->lchild=p;
		}
	return bt;
}

BiTree InsertR(BiTree bt,char x,BiTree parent){
	BiTree p;
	if(parent==NULL){
		cout<<"�������"<<endl;
		return NULL;
	}
	p=new BiTNode;
	if(!p)
		return NULL;
	p->data=x;
	p->lchild=NULL;
	p->rchild=NULL;
	if(parent->rchild==NULL)
		parent->rchild=p;
	else{
			p->rchild=parent->rchild;
			parent->rchild=p;
		}
	return bt;
}
void main()
{
    cout<<"Blue Hope Studio (C) Copyright 2000-2007"<<endl;
    cout<<"Wang Xun's Data Struct Work."<<endl;
    BiTree a=new BiTNode;
    Initiate(a);
	a->data='A';
    BiTNode *b,*c,*d,*e;
    InsertL(a,'B',a);
    InsertR(a,'C',a);
     b=a->lchild; 
    InsertL(a,'D',b);
    InsertR(a,'E',b);

	//�����ǲ��Դ���...
	cout<<a->data<<endl;
	cout<<a->lchild->data<<endl;
	cout<<a->rchild->data<<endl;
	cout<<a->lchild->lchild->data<<endl;
	cout<<a->lchild->rchild->data<<endl;
	cout<<a->rchild->lchild<<"(����)"<<endl; // ����
	//���Դ������...

    cout<<"Run Finished..."<<endl;
}