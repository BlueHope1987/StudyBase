//GBK���뱣���Է�ֹ"�������л��з�"����
/*
http://c.biancheng.net/view/1320.html
GoF �� 23 �����ģʽ�ķ���͹���

1. ����Ŀ������
����ģʽ���������ʲô���������֣����ַ�ʽ�ɷ�Ϊ������ģʽ���ṹ��ģʽ����Ϊ��ģʽ 3 �֡�
������ģʽ�����������������������󡱣�������Ҫ�ص��ǡ�������Ĵ�����ʹ�÷��롱��GoF ���ṩ�˵�����ԭ�͡��������������󹤳��������ߵ� 5 �ִ�����ģʽ��
�ṹ��ģʽ������������ν�������ĳ�ֲ�����ɸ���Ľṹ��GoF ���ṩ�˴������������Žӡ�װ�Ρ���ۡ���Ԫ����ϵ� 7 �ֽṹ��ģʽ��
��Ϊ��ģʽ����������������֮�������໥Э����ͬ��ɵ��������޷�������ɵ������Լ���������ְ��GoF ���ṩ��ģ�巽�������ԡ����ְ������״̬���۲��ߡ��н��ߡ��������������ߡ�����¼���������� 11 ����Ϊ��ģʽ��
2. �������÷�Χ����
����ģʽ����Ҫ�������ϻ�����Ҫ���ڶ��������֣����ַ�ʽ�ɷ�Ϊ��ģʽ�Ͷ���ģʽ���֡�
��ģʽ�����ڴ�����������֮��Ĺ�ϵ����Щ��ϵͨ���̳����������Ǿ�̬�ģ��ڱ���ʱ�̱�ȷ�������ˡ�GoF�еĹ������������ࣩ��������ģ�巽�������������ڸ�ģʽ��
����ģʽ�����ڴ������֮��Ĺ�ϵ����Щ��ϵ����ͨ����ϻ�ۺ���ʵ�֣�������ʱ���ǿ��Ա仯�ģ����߶�̬�ԡ�GoF �г������� 4 �֣������Ķ��Ƕ���ģʽ��

�� 1 �������� 23 �����ģʽ�ķ��ࡣ

��1GoF �� 23 �����ģʽ�ķ����
��Χ\Ŀ��   ������ģʽ  �ṹ��ģʽ    ��Ϊ��ģʽ

��ģʽ  ��������    (�ࣩ������    ģ�巽����������

����ģʽ	����	    ����          ����
			ԭ��   (����������      ����
		   ���󹤳�       �Ž�       ְ����
		  ������          װ��        ״̬
						 ���        �۲���
						 ��Ԫ        �н���
						 ���        ������
									������
									����¼

3. GoF��23�����ģʽ�Ĺ���
ǰ��˵���� GoF �� 23 �����ģʽ�ķ��࣬���ڶԸ���ģʽ�Ĺ��ܽ��н��ܡ�
������Singleton��ģʽ��ĳ����ֻ������һ��ʵ���������ṩ��һ��ȫ�ַ��ʵ㹩�ⲿ��ȡ��ʵ��������չ�����޶���ģʽ��
ԭ�ͣ�Prototype��ģʽ����һ��������Ϊԭ�ͣ�ͨ��������и��ƶ���¡�������ԭ�����Ƶ���ʵ����
����������Factory Method��ģʽ������һ�����ڴ�����Ʒ�Ľӿڣ��������������ʲô��Ʒ��
���󹤳���AbstractFactory��ģʽ���ṩһ��������Ʒ��Ľӿڣ���ÿ�������������һϵ����صĲ�Ʒ��
�����ߣ�Builder��ģʽ����һ�����Ӷ���ֽ�ɶ����Լ򵥵Ĳ��֣�Ȼ����ݲ�ͬ��Ҫ�ֱ𴴽����ǣ���󹹽��ɸø��Ӷ���
����Proxy��ģʽ��Ϊĳ�����ṩһ�ִ����Կ��ƶԸö���ķ��ʡ����ͻ���ͨ�������ӵط��ʸö��󣬴Ӷ����ơ���ǿ���޸ĸö����һЩ���ԡ�
��������Adapter��ģʽ����һ����Ľӿ�ת���ɿͻ�ϣ��������һ���ӿڣ�ʹ��ԭ�����ڽӿڲ����ݶ�����һ��������Щ����һ������
�Žӣ�Bridge��ģʽ����������ʵ�ַ��룬ʹ���ǿ��Զ����仯����������Ϲ�ϵ����̳й�ϵ��ʵ�֣��Ӷ������˳����ʵ���������ɱ�ά�ȵ���϶ȡ�
װ�Σ�Decorator��ģʽ����̬�ĸ���������һЩְ�𣬼����������Ĺ��ܡ�
��ۣ�Facade��ģʽ��Ϊ������ӵ���ϵͳ�ṩһ��һ�µĽӿڣ�ʹ��Щ��ϵͳ�������ױ����ʡ�
��Ԫ��Flyweight��ģʽ�����ù���������Ч��֧�ִ���ϸ���ȶ���ĸ��á�
��ϣ�Composite��ģʽ����������ϳ���״��νṹ��ʹ�û��Ե����������϶������һ�µķ����ԡ�
ģ�巽����TemplateMethod��ģʽ������һ�������е��㷨�Ǽܣ������㷨��һЩ�����ӳٵ������У�ʹ��������Բ��ı���㷨�ṹ��������ض�����㷨��ĳЩ�ض����衣
���ԣ�Strategy��ģʽ��������һϵ���㷨������ÿ���㷨��װ������ʹ���ǿ����໥�滻�����㷨�ĸı䲻��Ӱ��ʹ���㷨�Ŀͻ���
���Command��ģʽ����һ�������װΪһ������ʹ������������κ�ִ����������ηָ��
ְ������Chain of Responsibility��ģʽ������������е�һ�����󴫵���һ������ֱ��������ӦΪֹ��ͨ�����ַ�ʽȥ������֮�����ϡ�
״̬��State��ģʽ������һ�����������ڲ�״̬�����ı�ʱ�ı�����Ϊ������
�۲��ߣ�Observer��ģʽ�������������һ�Զ��ϵ����һ���������ı�ʱ�������ָı�֪ͨ������������󣬴Ӷ�Ӱ�������������Ϊ��
�н��ߣ�Mediator��ģʽ������һ���н��������ԭ�ж���֮��Ľ�����ϵ������ϵͳ�ж�������϶ȣ�ʹԭ�ж���֮�䲻���໥�˽⡣
��������Iterator��ģʽ���ṩһ�ַ�����˳����ʾۺ϶����е�һϵ�����ݣ�������¶�ۺ϶�����ڲ���ʾ��
�����ߣ�Visitor��ģʽ���ڲ��ı伯��Ԫ�ص�ǰ���£�Ϊһ�������е�ÿ��Ԫ���ṩ���ַ��ʷ�ʽ����ÿ��Ԫ���ж�������߶�����ʡ�
����¼��Memento��ģʽ���ڲ��ƻ���װ�Ե�ǰ���£���ȡ������һ��������ڲ�״̬���Ա��Ժ�ָ�����
��������Interpreter��ģʽ���ṩ��ζ������Ե��ķ����Լ������Ծ��ӵĽ��ͷ���������������

*/
//https://blog.csdn.net/CoderAldrich/article/details/83272866

#include <iostream>

#include <map>
#include <vector>

#include <string>

#include <list>

using namespace std;

//1������ģʽ
namespace sng {
	class Singleton
	{
	public:
		static Singleton *GetInstance()
		{
			if (m_Instance == NULL)
			{
				m_Instance = new Singleton();
			}
			return m_Instance;
		}

		static void DestoryInstance()
		{
			if (m_Instance != NULL)
			{
				delete m_Instance;
				m_Instance = NULL;
			}
		}

		// This is just a operation example
		int GetTest()
		{
			return m_Test;
		}

	private:
		Singleton() { m_Test = 10; }
		static Singleton *m_Instance;
		int m_Test;
	};
	Singleton *Singleton::m_Instance = NULL;
}
void sngTest() {
	//����ģʽ����
	sng::Singleton *singletonObj = sng::Singleton::GetInstance();
	cout << singletonObj->GetTest() << endl;
	sng::Singleton::DestoryInstance();
	return;
};

//2��ԭ��ģʽ
namespace ptp {
	class Prototype
	{
	public:
		Prototype() {}
		virtual ~Prototype() {}

		virtual Prototype * Clone() = 0;
	};
	class ConcretePrototype : public Prototype
	{
	public:
		ConcretePrototype() :m_counter(0) {}
		virtual ~ConcretePrototype() {}

		//�������캯��
		ConcretePrototype(const ConcretePrototype & rhs)
		{
			m_counter = rhs.m_counter;
		}

		//��������
		virtual ConcretePrototype * Clone()
		{
			//���ÿ������캯��
			return new ConcretePrototype(*this);
		}

	private:
		int m_counter;
	};
}
void ptpTest() {
	//ԭ��ģʽ����
	ptp::ConcretePrototype * conProA = new ptp::ConcretePrototype();
	ptp::ConcretePrototype * conProB = conProA->Clone();
	delete conProA;
	conProA = NULL;
	delete conProB;
	conProB = NULL;
	return;
};

//3����������
namespace fac {
	class Product
	{
	public:
		virtual void Show() = 0;
	};

	class ProductA : public Product
	{
	public:
		void Show()
		{
			cout << "I'm ProductA" << endl;
		}
	};

	class ProductB : public Product
	{
	public:
		void Show()
		{
			cout << "I'm ProductB" << endl;
		}
	};

	class Factory
	{
	public:
		virtual Product *CreateProduct() = 0;
	};

	class FactoryA : public Factory
	{
	public:
		Product *CreateProduct()
		{
			return new ProductA();
		}
	};

	class FactoryB : public Factory
	{
	public:
		Product *CreateProduct()
		{
			return new ProductB();
		}
	};
}
void facTest() {
	//������������
	fac::Factory *factoryA = new fac::FactoryA();
	fac::Product *productA = factoryA->CreateProduct();
	productA->Show();
	fac::Factory *factoryB = new fac::FactoryB();
	fac::Product *productB = factoryB->CreateProduct();
	productB->Show();
	if (factoryA != NULL)
	{
		delete factoryA;
		factoryA = NULL;
	}
	if (productA != NULL)
	{
		delete productA;
		productA = NULL;
	}
	if (factoryB != NULL)
	{
		delete factoryB;
		factoryB = NULL;
	}
	if (productB != NULL)
	{
		delete productB;
		productB = NULL;
	}
	return;
};

//4�����󹤳�
namespace absfac {
	class ProductA
	{
	public:
		virtual void Show() = 0;
	};

	class ProductA1 : public ProductA
	{
	public:
		void Show()
		{
			cout << "I'm ProductA1" << endl;
		}
	};

	class ProductA2 : public ProductA
	{
	public:
		void Show()
		{
			cout << "I'm ProductA2" << endl;
		}
	};

	// Product B
	class ProductB
	{
	public:
		virtual void Show() = 0;
	};

	class ProductB1 : public ProductB
	{
	public:
		void Show()
		{
			cout << "I'm ProductB1" << endl;
		}
	};

	class ProductB2 : public ProductB
	{
	public:
		void Show()
		{
			cout << "I'm ProductB2" << endl;
		}
	};

	// Factory
	class Factory
	{
	public:
		virtual ProductA *CreateProductA() = 0;
		virtual ProductB *CreateProductB() = 0;
	};

	class Factory1 : public Factory
	{
	public:
		ProductA *CreateProductA()
		{
			return new ProductA1();
		}

		ProductB *CreateProductB()
		{
			return new ProductB1();
		}
	};

	class Factory2 : public Factory
	{
		ProductA *CreateProductA()
		{
			return new ProductA2();
		}

		ProductB *CreateProductB()
		{
			return new ProductB2();
		}
	};
}
void absfacTest() {
	//���󹤳�����
	absfac::Factory *factoryObj1 = new absfac::Factory1();
	absfac::ProductA *productObjA1 = factoryObj1->CreateProductA();
	absfac::ProductB *productObjB1 = factoryObj1->CreateProductB();

	productObjA1->Show();
	productObjB1->Show();

	absfac::Factory *factoryObj2 = new absfac::Factory2();
	absfac::ProductA *productObjA2 = factoryObj2->CreateProductA();
	absfac::ProductB *productObjB2 = factoryObj2->CreateProductB();

	productObjA2->Show();
	productObjB2->Show();

	if (factoryObj1 != NULL)
	{
		delete factoryObj1;
		factoryObj1 = NULL;
	}

	if (productObjA1 != NULL)
	{
		delete productObjA1;
		productObjA1 = NULL;
	}

	if (productObjB1 != NULL)
	{
		delete productObjB1;
		productObjB1 = NULL;
	}

	if (factoryObj2 != NULL)
	{
		delete factoryObj2;
		factoryObj2 = NULL;
	}

	if (productObjA2 != NULL)
	{
		delete productObjA2;
		productObjA2 = NULL;
	}

	if (productObjB2 != NULL)
	{
		delete productObjB2;
		productObjB2 = NULL;
	}
	return;
};

//5��������
namespace bdr {
	typedef enum MANTYPETag
	{
		kFatMan,
		kThinMan,
		kNormal
	}MANTYPE;

	class Man
	{
	public:
		void SetHead(MANTYPE type) { m_Type = type; }
		void SetBody(MANTYPE type) { m_Type = type; }
		void SetLeftHand(MANTYPE type) { m_Type = type; }
		void SetRightHand(MANTYPE type) { m_Type = type; }
		void SetLeftFoot(MANTYPE type) { m_Type = type; }
		void SetRightFoot(MANTYPE type) { m_Type = type; }
		void ShowMan()
		{
			switch (m_Type)
			{
			case kFatMan:
				cout << "I'm a fat man" << endl;
				return;

			case kThinMan:
				cout << "I'm a thin man" << endl;
				return;

			default:
				cout << "I'm a normal man" << endl;
				return;
			}
		}

	private:
		MANTYPE m_Type;
	};

	// Builder
	class Builder
	{
	public:
		virtual void BuildHead() {}
		virtual void BuildBody() {}
		virtual void BuildLeftHand() {}
		virtual void BuildRightHand() {}
		virtual void BuildLeftFoot() {}
		virtual void BuildRightFoot() {}
		virtual Man *GetMan() { return NULL; }
	};

	// FatManBuilder
	class FatManBuilder : public Builder
	{
	public:
		FatManBuilder() { m_FatMan = new Man(); }
		void BuildHead() { m_FatMan->SetHead(kFatMan); }
		void BuildBody() { m_FatMan->SetBody(kFatMan); }
		void BuildLeftHand() { m_FatMan->SetLeftHand(kFatMan); }
		void BuildRightHand() { m_FatMan->SetRightHand(kFatMan); }
		void BuildLeftFoot() { m_FatMan->SetLeftFoot(kFatMan); }
		void BuildRightFoot() { m_FatMan->SetRightFoot(kFatMan); }
		Man *GetMan() { return m_FatMan; }

	private:
		Man *m_FatMan;
	};

	// ThisManBuilder
	class ThinManBuilder : public Builder
	{
	public:
		ThinManBuilder() { m_ThinMan = new Man(); }
		void BuildHead() { m_ThinMan->SetHead(kThinMan); }
		void BuildBody() { m_ThinMan->SetBody(kThinMan); }
		void BuildLeftHand() { m_ThinMan->SetLeftHand(kThinMan); }
		void BuildRightHand() { m_ThinMan->SetRightHand(kThinMan); }
		void BuildLeftFoot() { m_ThinMan->SetLeftFoot(kThinMan); }
		void BuildRightFoot() { m_ThinMan->SetRightFoot(kThinMan); }
		Man *GetMan() { return m_ThinMan; }

	private:
		Man *m_ThinMan;
	};

	// Director
	class Director
	{
	public:
		Director(Builder *builder) { m_Builder = builder; }
		void CreateMan();

	private:
		Builder *m_Builder;
	};

	void Director::CreateMan()
	{
		m_Builder->BuildHead();
		m_Builder->BuildBody();
		m_Builder->BuildLeftHand();
		m_Builder->BuildRightHand();
		m_Builder->BuildLeftHand();
		m_Builder->BuildRightHand();
	}
}
void bdrTest() {
	//�����߲���
	bdr::Builder *builderObj = new bdr::FatManBuilder();
	bdr::Director directorObj(builderObj);
	directorObj.CreateMan();
	bdr::Man *manObj = builderObj->GetMan();
	if (manObj == NULL)
		return;
	manObj->ShowMan();
	delete manObj;
	manObj = NULL;
	delete builderObj;
	builderObj = NULL;
	return;
};

//6������ģʽ
namespace prx {
#define SAFE_DELETE(p) if (p) { delete p; p = NULL;}

	class CSubject
	{
	public:
		CSubject() {};
		virtual ~CSubject() {}

		virtual void Request() = 0;
	};

	class CRealSubject : public CSubject
	{
	public:
		CRealSubject() {}
		~CRealSubject() {}
		void Request()
		{
			cout << "CRealSubject Request" << endl;
		}
	};

	class CProxy : public CSubject
	{
	public:
		CProxy() : m_pRealSubject(NULL) {}
		~CProxy()
		{
			SAFE_DELETE(m_pRealSubject);
		}

		void Request()
		{
			if (NULL == m_pRealSubject)
			{
				m_pRealSubject = new CRealSubject();
			}
			cout << "CProxy Request" << endl;
			m_pRealSubject->Request();
		}

	private:
		CRealSubject *m_pRealSubject;
	};

}
void prxTest() {
	//����ģʽ����
	prx::CSubject *pSubject = new prx::CProxy();
	pSubject->Request();
	SAFE_DELETE(pSubject);
	return;
};

//7��������ģʽ
namespace adp {
	// Targets
	class Target
	{
	public:
		virtual void Request()
		{
			cout << "Target::Request" << endl;
		}
	};

	// Adaptee
	class Adaptee
	{
	public:
		void SpecificRequest()
		{
			cout << "Adaptee::SpecificRequest" << endl;
		}
	};

	// Adapter
	class Adapter : public Target, Adaptee
	{
	public:
		void Request()
		{
			Adaptee::SpecificRequest();
		}
	};
}
void adpTest() {
	//����������
	adp::Target *targetObj = new adp::Adapter();
	targetObj->Request();
	delete targetObj;
	targetObj = NULL;
	return;
};

//8���Ž�ģʽ
namespace bdg {
	class Implementor
	{
	public:
		virtual void OperationImpl() = 0;
	};
	class ConcreteImpementor : public Implementor
	{
	public:
		void OperationImpl()
		{
			cout << "OperationImpl" << endl;
		}
	};
	class Abstraction
	{
	public:
		Abstraction(Implementor *pImpl) : m_pImpl(pImpl) {}
		virtual void Operation() = 0;

	protected:
		Implementor *m_pImpl;
	};
	class RedfinedAbstraction : public Abstraction
	{
	public:
		RedfinedAbstraction(Implementor *pImpl) : Abstraction(pImpl) {}
		void Operation()
		{
			m_pImpl->OperationImpl();
		}
	};
}
void bdgTest() {
	//�Ž�ģʽ����
	bdg::Implementor *pImplObj = new bdg::ConcreteImpementor();
	bdg::Abstraction *pAbsObj = new bdg::RedfinedAbstraction(pImplObj);
	pAbsObj->Operation();
	delete pImplObj;
	pImplObj = NULL;
	delete pAbsObj;
	pAbsObj = NULL;
	return;
};

//9��װ��ģʽ
namespace dec {
	class Component
	{
	public:
		virtual void Operation() = 0;
	};
	class ConcreteComponent : public Component
	{
	public:
		void Operation()
		{
			cout << "I am no decoratored ConcreteComponent" << endl;
		}
	};
	class Decorator : public Component
	{
	public:
		Decorator(Component *pComponent) : m_pComponentObj(pComponent) {}
		void Operation()
		{
			if (m_pComponentObj != NULL)
			{
				m_pComponentObj->Operation();
			}
		}
	protected:
		Component *m_pComponentObj;
	};
	class ConcreteDecoratorA : public Decorator
	{
	public:
		ConcreteDecoratorA(Component *pDecorator) : Decorator(pDecorator) {}
		void Operation()
		{
			AddedBehavior();
			Decorator::Operation();
		}
		void  AddedBehavior()
		{
			cout << "This is added behavior A." << endl;
		}
	};
	class ConcreteDecoratorB : public Decorator
	{
	public:
		ConcreteDecoratorB(Component *pDecorator) : Decorator(pDecorator) {}
		void Operation()
		{
			AddedBehavior();
			Decorator::Operation();
		}
		void  AddedBehavior()
		{
			cout << "This is added behavior B." << endl;
		}
	};
}
void decTest() {
	//װ��ģʽ����
	dec::Component *pComponentObj = new dec::ConcreteComponent();
	dec::Decorator *pDecoratorAOjb = new dec::ConcreteDecoratorA(pComponentObj);
	pDecoratorAOjb->Operation();
	cout << "=============================================" << endl;
	dec::Decorator *pDecoratorBOjb = new dec::ConcreteDecoratorB(pComponentObj);
	pDecoratorBOjb->Operation();
	cout << "=============================================" << endl;
	dec::Decorator *pDecoratorBAOjb = new dec::ConcreteDecoratorB(pDecoratorAOjb);
	pDecoratorBAOjb->Operation();
	cout << "=============================================" << endl;
	delete pDecoratorBAOjb;
	pDecoratorBAOjb = NULL;
	delete pDecoratorBOjb;
	pDecoratorBOjb = NULL;
	delete pDecoratorAOjb;
	pDecoratorAOjb = NULL;
	delete pComponentObj;
	pComponentObj = NULL;
	return;
};

//10�����ģʽ
namespace fcd {
	// �﷨������ϵͳ
	class CSyntaxParser
	{
	public:
		void SyntaxParser()
		{
			cout << "Syntax Parser" << endl;
		}
	};

	// �����м������ϵͳ
	class CGenMidCode
	{
	public:
		void GenMidCode()
		{
			cout << "Generate middle code" << endl;
		}
	};

	// ���ɻ�������ϵͳ
	class CGenAssemblyCode
	{
	public:
		void GenAssemblyCode()
		{
			cout << "Generate assembly code" << endl;
		}
	};

	// �������ɿ�ִ��Ӧ�ó�������ϵͳ
	class CLinkSystem
	{
	public:
		void LinkSystem()
		{
			cout << "Link System" << endl;
		}
	};

	class Facade
	{
	public:
		void Compile()
		{
			CSyntaxParser syntaxParser;
			CGenMidCode genMidCode;
			CGenAssemblyCode genAssemblyCode;
			CLinkSystem linkSystem;
			syntaxParser.SyntaxParser();
			genMidCode.GenMidCode();
			genAssemblyCode.GenAssemblyCode();
			linkSystem.LinkSystem();
		}
	};
}
void fcdTest() {
	fcd::Facade facade;
	facade.Compile();
	return;
}

//11����Ԫģʽ
namespace fyw {
	typedef struct pointTag
	{
		int x;
		int y;

		pointTag() {}
		pointTag(int a, int b)
		{
			x = a;
			y = b;
		}

		bool operator <(const pointTag& other) const
		{
			if (x < other.x)
			{
				return true;
			}
			else if (x == other.x)
			{
				return y < other.y;
			}

			return false;
		}
	}POINT;

	typedef enum PieceColorTag
	{
		BLACK,
		WHITE
	}PIECECOLOR;

	class CPiece
	{
	public:
		CPiece(PIECECOLOR color) : m_color(color) {}
		PIECECOLOR GetColor() { return m_color; }

		// Set the external state
		void SetPoint(POINT point) { m_point = point; }
		POINT GetPoint() { return m_point; }

	protected:
		// Internal state
		PIECECOLOR m_color;

		// external state
		POINT m_point;
	};

	class CGomoku : public CPiece
	{
	public:
		CGomoku(PIECECOLOR color) : CPiece(color) {}
	};

	class CPieceFactory
	{
	public:
		CPiece *GetPiece(PIECECOLOR color)
		{
			CPiece *pPiece = NULL;
			if (m_vecPiece.empty())
			{
				pPiece = new CGomoku(color);
				m_vecPiece.push_back(pPiece);
			}
			else
			{
				for (vector<CPiece *>::iterator it = m_vecPiece.begin(); it != m_vecPiece.end(); ++it)
				{
					if ((*it)->GetColor() == color)
					{
						pPiece = *it;
						break;
					}
				}
				if (pPiece == NULL)
				{
					pPiece = new CGomoku(color);
					m_vecPiece.push_back(pPiece);
				}
			}
			return pPiece;
		}

		~CPieceFactory()
		{
			for (vector<CPiece *>::iterator it = m_vecPiece.begin(); it != m_vecPiece.end(); ++it)
			{
				if (*it != NULL)
				{
					delete *it;
					*it = NULL;
				}
			}
		}

	private:
		vector<CPiece *> m_vecPiece;
	};

	class CChessboard
	{
	public:
		void Draw(CPiece *piece)
		{
			if (piece->GetColor())
			{
				cout << "Draw a White" << " at (" << piece->GetPoint().x << "," << piece->GetPoint().y << ")" << endl;
			}
			else
			{
				cout << "Draw a Black" << " at (" << piece->GetPoint().x << "," << piece->GetPoint().y << ")" << endl;
			}
			m_mapPieces.insert(pair<POINT, CPiece *>(piece->GetPoint(), piece));
		}

		void ShowAllPieces()
		{
			for (map<POINT, CPiece *>::iterator it = m_mapPieces.begin(); it != m_mapPieces.end(); ++it)
			{
				if (it->second->GetColor())
				{
					cout << "(" << it->first.x << "," << it->first.y << ") has a White chese." << endl;
				}
				else
				{
					cout << "(" << it->first.x << "," << it->first.y << ") has a Black chese." << endl;
				}
			}
		}

	private:
		map<POINT, CPiece *> m_mapPieces;
	};
}
void fywTest() {
	fyw::CPieceFactory *pPieceFactory = new fyw::CPieceFactory();
	fyw::CChessboard *pCheseboard = new fyw::CChessboard();

	// The player1 get a white piece from the pieces bowl
	fyw::CPiece *pPiece = pPieceFactory->GetPiece(fyw::WHITE);
	pPiece->SetPoint(fyw::POINT(2, 3));
	pCheseboard->Draw(pPiece);

	// The player2 get a black piece from the pieces bowl
	pPiece = pPieceFactory->GetPiece(fyw::BLACK);
	pPiece->SetPoint(fyw::POINT(4, 5));
	pCheseboard->Draw(pPiece);

	// The player1 get a white piece from the pieces bowl
	pPiece = pPieceFactory->GetPiece(fyw::WHITE);
	pPiece->SetPoint(fyw::POINT(2, 4));
	pCheseboard->Draw(pPiece);

	// The player2 get a black piece from the pieces bowl
	pPiece = pPieceFactory->GetPiece(fyw::BLACK);
	pPiece->SetPoint(fyw::POINT(3, 5));
	pCheseboard->Draw(pPiece);

	/*......*/

	//Show all cheses
	cout << "Show all cheses" << endl;
	pCheseboard->ShowAllPieces();

	if (pCheseboard != NULL)
	{
		delete pCheseboard;
		pCheseboard = NULL;
	}
	if (pPieceFactory != NULL)
	{
		delete pPieceFactory;
		pPieceFactory = NULL;
	}
}

//12�����ģʽ
namespace cmp {
	class Component
	{
	public:
		Component(string name) : m_strCompname(name) {}
		virtual ~Component() {}
		virtual void Operation() = 0;
		virtual void Add(Component *) = 0;
		virtual void Remove(Component *) = 0;
		virtual Component *GetChild(int) = 0;
		virtual string GetName()
		{
			return m_strCompname;
		}
		virtual void Print() = 0;
	protected:
		string m_strCompname;
	};
	class Leaf : public Component
	{
	public:
		Leaf(string name) : Component(name)
		{}
		void Operation()
		{
			cout << "I'm " << m_strCompname << endl;
		}
		void Add(Component *pComponent) {}
		void Remove(Component *pComponent) {}
		Component *GetChild(int index)
		{
			return NULL;
		}
		void Print() {}
	};
	class Composite : public Component
	{
	public:
		Composite(string name) : Component(name)
		{}
		~Composite()
		{
			vector<Component *>::iterator it = m_vecComp.begin();
			while (it != m_vecComp.end())
			{
				if (*it != NULL)
				{
					cout << "----delete " << (*it)->GetName() << "----" << endl;
					delete *it;
					*it = NULL;
				}
				m_vecComp.erase(it);
				it = m_vecComp.begin();
			}
		}
		void Operation()
		{
			cout << "I'm " << m_strCompname << endl;
		}
		void Add(Component *pComponent)
		{
			m_vecComp.push_back(pComponent);
		}
		void Remove(Component *pComponent)
		{
			for (vector<Component *>::iterator it = m_vecComp.begin(); it != m_vecComp.end(); ++it)
			{
				if ((*it)->GetName() == pComponent->GetName())
				{
					if (*it != NULL)
					{
						delete *it;
						*it = NULL;
					}
					m_vecComp.erase(it);
					break;
				}
			}
		}
		Component *GetChild(int index)
		{
			if (index > m_vecComp.size())
			{
				return NULL;
			}
			return m_vecComp[index - 1];
		}
		void Print()
		{
			for (vector<Component *>::iterator it = m_vecComp.begin(); it != m_vecComp.end(); ++it)
			{
				cout << (*it)->GetName() << endl;
			}
		}
	private:
		vector<Component *> m_vecComp;
	};
}
void cmpTest() {
	cmp::Component *pNode = new cmp::Composite("Beijing Head Office");
	cmp::Component *pNodeHr = new cmp::Leaf("Beijing Human Resources Department");
	cmp::Component *pSubNodeSh = new cmp::Composite("Shanghai Branch");
	cmp::Component *pSubNodeCd = new cmp::Composite("Chengdu Branch");
	cmp::Component *pSubNodeBt = new cmp::Composite("Baotou Branch");
	pNode->Add(pNodeHr);
	pNode->Add(pSubNodeSh);
	pNode->Add(pSubNodeCd);
	pNode->Add(pSubNodeBt);
	pNode->Print();
	cmp::Component *pSubNodeShHr = new cmp::Leaf("Shanghai Human Resources Department");
	cmp::Component *pSubNodeShCg = new cmp::Leaf("Shanghai Purchasing Department");
	cmp::Component *pSubNodeShXs = new cmp::Leaf("Shanghai Sales department");
	cmp::Component *pSubNodeShZb = new cmp::Leaf("Shanghai Quality supervision Department");
	pSubNodeSh->Add(pSubNodeShHr);
	pSubNodeSh->Add(pSubNodeShCg);
	pSubNodeSh->Add(pSubNodeShXs);
	pSubNodeSh->Add(pSubNodeShZb);
	pNode->Print();
	// ��˾����������Ҫ�ر��Ϻ������ල����
	pSubNodeSh->Remove(pSubNodeShZb);
	if (pNode != NULL)
	{
		delete pNode;
		pNode = NULL;
	}
	return;
}

//13��ģ�巽��
namespace tmp {
	class AbstractClass
	{
	public:
		void TemplateMethod()
		{
			PrimitiveOperation1();
			cout << "TemplateMethod" << endl;
			PrimitiveOperation2();
		}

	protected:
		virtual void PrimitiveOperation1()
		{
			cout << "Default Operation1" << endl;
		}

		virtual void PrimitiveOperation2()
		{
			cout << "Default Operation2" << endl;
		}
	};

	class ConcreteClassA : public AbstractClass
	{
	protected:
		virtual void PrimitiveOperation1()
		{
			cout << "ConcreteA Operation1" << endl;
		}

		virtual void PrimitiveOperation2()
		{
			cout << "ConcreteA Operation2" << endl;
		}
	};

	class ConcreteClassB : public AbstractClass
	{
	protected:
		virtual void PrimitiveOperation1()
		{
			cout << "ConcreteB Operation1" << endl;
		}

		virtual void PrimitiveOperation2()
		{
			cout << "ConcreteB Operation2" << endl;
		}
	};
}
void tmpTest() {
	tmp::AbstractClass *pAbstractA = new tmp::ConcreteClassA;
	pAbstractA->TemplateMethod();

	tmp::AbstractClass *pAbstractB = new tmp::ConcreteClassB;
	pAbstractB->TemplateMethod();

	if (pAbstractA) delete pAbstractA;
	if (pAbstractB) delete pAbstractB;
	return;
}

//14������ģʽ
namespace stg {
	// The abstract strategy
	class Strategy
	{
	public:
		virtual void AlgorithmInterface() = 0;
	};

	class ConcreteStrategyA : public Strategy
	{
	public:
		void AlgorithmInterface()
		{
			cout << "I am from ConcreteStrategyA." << endl;
		}
	};

	class ConcreteStrategyB : public Strategy
	{
	public:
		void AlgorithmInterface()
		{
			cout << "I am from ConcreteStrategyB." << endl;
		}
	};

	class ConcreteStrategyC : public Strategy
	{
	public:
		void AlgorithmInterface()
		{
			cout << "I am from ConcreteStrategyC." << endl;
		}
	};

	class Context
	{
	public:
		Context(Strategy *pStrategyArg) : pStrategy(pStrategyArg)
		{
		}
		void ContextInterface()
		{
			pStrategy->AlgorithmInterface();
		}
	private:
		Strategy *pStrategy;
	};
}
void stgTest() {
	// Create the Strategy
	stg::Strategy *pStrategyA = new stg::ConcreteStrategyA;
	stg::Strategy *pStrategyB = new stg::ConcreteStrategyB;
	stg::Strategy *pStrategyC = new stg::ConcreteStrategyC;
	stg::Context *pContextA = new stg::Context(pStrategyA);
	stg::Context *pContextB = new stg::Context(pStrategyB);
	stg::Context *pContextC = new stg::Context(pStrategyC);
	pContextA->ContextInterface();
	pContextB->ContextInterface();
	pContextC->ContextInterface();

	if (pStrategyA) delete pStrategyA;
	if (pStrategyB) delete pStrategyB;
	if (pStrategyC) delete pStrategyC;

	if (pContextA) delete pContextA;
	if (pContextB) delete pContextB;
	if (pContextC) delete pContextC;
	return;
};

//15������ģʽ
namespace cmd {
#define SAFE_DELETE(p) if (p) { delete p; p = NULL; }

	class Receiver
	{
	public:
		void Action()
		{
			cout << "Receiver->Action" << endl;
		}
	};

	class Command
	{
	public:
		virtual void Execute() = 0;
	};

	class ConcreteCommand : public Command
	{
	public:
		ConcreteCommand(Receiver *pReceiver) : m_pReceiver(pReceiver) {}
		void Execute()
		{
			m_pReceiver->Action();
		}
	private:
		Receiver *m_pReceiver;
	};

	class Invoker
	{
	public:
		Invoker(Command *pCommand) : m_pCommand(pCommand) {}
		void Invoke()
		{
			m_pCommand->Execute();
		}
	private:
		Command *m_pCommand;
	};
};
void cmdTest() {
	cmd::Receiver *pReceiver = new cmd::Receiver();
	cmd::Command *pCommand = new cmd::ConcreteCommand(pReceiver);
	cmd::Invoker *pInvoker = new cmd::Invoker(pCommand);
	pInvoker->Invoke();
	SAFE_DELETE(pInvoker);
	SAFE_DELETE(pCommand);
	SAFE_DELETE(pReceiver);
	return;
}

//16��ְ����ģʽ
namespace cor {
#define SAFE_DELETE(p) if (p) { delete p; p = NULL; }

	class HolidayRequest
	{
	public:
		HolidayRequest(int hour) : m_iHour(hour) {}
		int GetHour() { return m_iHour; }
	private:
		int m_iHour;
	};

	// The holiday request handler interface
	class Manager
	{
	public:
		virtual bool HandleRequest(HolidayRequest *pRequest) = 0;
	};

	// Project manager
	class PM : public Manager
	{
	public:
		PM(Manager *handler) : m_pHandler(handler) {}
		bool HandleRequest(HolidayRequest *pRequest)
		{
			if (pRequest->GetHour() <= 2 || m_pHandler == NULL)
			{
				cout << "PM said:OK." << endl;
				return true;
			}
			return m_pHandler->HandleRequest(pRequest);
		}
	private:
		Manager *m_pHandler;
	};

	// Department manager
	class DM : public Manager
	{
	public:
		DM(Manager *handler) : m_pHandler(handler) {}
		bool HandleRequest(HolidayRequest *pRequest)
		{
			cout << "DM said:OK." << endl;
			return true;
		}

		// The department manager is in?
		bool IsIn()
		{
			return true;
		}
	private:
		Manager *m_pHandler;
	};

	// Project supervisor
	class PS : public Manager
	{
	public:
		PS(Manager *handler) : m_pHandler(handler) {}
		bool HandleRequest(HolidayRequest *pRequest)
		{
			DM *pDM = dynamic_cast<DM *>(m_pHandler);
			if (pDM != NULL)
			{
				if (pDM->IsIn())
				{
					return pDM->HandleRequest(pRequest);
				}
			}
			cout << "PS said:OK." << endl;
			return true;
		}
	private:
		Manager *m_pHandler;
	};
}
void corTest() {
	cor::DM *pDM = new cor::DM(NULL);
	cor::PS *pPS = new cor::PS(pDM);
	cor::PM *pPM = new cor::PM(pPS);
	cor::HolidayRequest *pHolidayRequest = new cor::HolidayRequest(10);
	pPM->HandleRequest(pHolidayRequest);
	SAFE_DELETE(pHolidayRequest);

	pHolidayRequest = new cor::HolidayRequest(2);
	pPM->HandleRequest(pHolidayRequest);

	SAFE_DELETE(pDM);
	SAFE_DELETE(pPS);
	SAFE_DELETE(pPM);
	SAFE_DELETE(pHolidayRequest);
	return;
}

//17��״̬ģʽ
namespace ste {
	class Context;

	class State
	{
	public:
		virtual void Handle(Context *pContext) = 0;
	};

	class ConcreteStateA : public State
	{
	public:
		virtual void Handle(Context *pContext)
		{
			cout << "I am concretestateA." << endl;
		}
	};

	class ConcreteStateB : public State
	{
	public:
		virtual void Handle(Context *pContext)
		{
			cout << "I am concretestateB." << endl;
		}
	};

	class Context
	{
	public:
		Context(State *pState) : m_pState(pState) {}

		void Request()
		{
			if (m_pState)
			{
				m_pState->Handle(this);
			}
		}

		void ChangeState(State *pState)
		{
			m_pState = pState;
		}

	private:
		State *m_pState;
	};
};
void steTest() {
	ste::State *pStateA = new ste::ConcreteStateA();
	ste::State *pStateB = new ste::ConcreteStateB();
	ste::Context *pContext = new ste::Context(pStateA);
	pContext->Request();

	pContext->ChangeState(pStateB);
	pContext->Request();

	delete pContext;
	delete pStateB;
	delete pStateA;
	return;
};

//18���۲���ģʽ
namespace obs {
	class Observer
	{
	public:
		virtual void Update(int) = 0;
	};

	class Subject
	{
	public:
		virtual void Attach(Observer *) = 0;
		virtual void Detach(Observer *) = 0;
		virtual void Notify() = 0;
	};

	class ConcreteObserver : public Observer
	{
	public:
		ConcreteObserver(Subject *pSubject) : m_pSubject(pSubject) {}

		void Update(int value)
		{
			cout << "ConcreteObserver get the update. New State:" << value << endl;
		}

	private:
		Subject *m_pSubject;
	};

	class ConcreteObserver2 : public Observer
	{
	public:
		ConcreteObserver2(Subject *pSubject) : m_pSubject(pSubject) {}

		void Update(int value)
		{
			cout << "ConcreteObserver2 get the update. New State:" << value << endl;
		}

	private:
		Subject *m_pSubject;
	};

	class ConcreteSubject : public Subject
	{
	public:
		void Attach(Observer *pObserver);
		void Detach(Observer *pObserver);
		void Notify();

		void SetState(int state)
		{
			m_iState = state;
		}

	private:
		std::list<Observer *> m_ObserverList;
		int m_iState;
	};

	void ConcreteSubject::Attach(Observer *pObserver)
	{
		m_ObserverList.push_back(pObserver);
	}

	void ConcreteSubject::Detach(Observer *pObserver)
	{
		m_ObserverList.remove(pObserver);
	}

	void ConcreteSubject::Notify()
	{
		std::list<Observer *>::iterator it = m_ObserverList.begin();
		while (it != m_ObserverList.end())
		{
			(*it)->Update(m_iState);
			++it;
		}
	}
};
void obsTest() {
	// Create Subject
	obs::ConcreteSubject *pSubject = new obs::ConcreteSubject();

	// Create Observer
	obs::Observer *pObserver = new obs::ConcreteObserver(pSubject);
	obs::Observer *pObserver2 = new obs::ConcreteObserver2(pSubject);

	// Change the state
	pSubject->SetState(2);

	// Register the observer
	pSubject->Attach(pObserver);
	pSubject->Attach(pObserver2);

	pSubject->Notify();

	// Unregister the observer
	pSubject->Detach(pObserver);

	pSubject->SetState(3);
	pSubject->Notify();

	delete pObserver;
	delete pObserver2;
	delete pSubject;

	return;
};


//�н���ģʽ
namespace mdi {
#define SAFE_DELETE(p) if (p) { delete p; p = NULL; }

	class Mediator;

	class Colleague
	{
	public:
		Colleague(Mediator *pMediator) : m_pMediator(pMediator) {}

		virtual void Send(wchar_t *message) = 0;

	protected:
		Mediator *m_pMediator;
	};

	class ConcreteColleague1 : public Colleague
	{
	public:
		ConcreteColleague1(Mediator *pMediator) : Colleague(pMediator) {}

		void Send(wchar_t *message);

		void Notify(wchar_t *message)
		{
			wcout << message << endl;
		}
	};

	class ConcreteColleague2 : public Colleague
	{
	public:
		ConcreteColleague2(Mediator *pMediator) : Colleague(pMediator) {}

		void Send(wchar_t *message);

		void Notify(wchar_t *message)
		{
			cout << "ConcreteColleague2 is handling the message." << endl;
			wcout << message << endl;
		}
	};

	class Mediator
	{
	public:
		virtual void Sent(wchar_t *message, Colleague *pColleague) = 0;
	};

	class ConcreteMediator : public Mediator
	{
	public:
		// The mediator forward the message
		void Sent(wchar_t *message, Colleague *pColleague)
		{
			ConcreteColleague1 *pConcreteColleague1 = dynamic_cast<ConcreteColleague1 *>(pColleague);
			if (pConcreteColleague1)
			{
				cout << "The message is from ConcreteColleague1. Now mediator forward it to ConcreteColleague2" << endl;
				if (m_pColleague2)
				{
					m_pColleague2->Notify(message);
				}
			}
			else
			{
				if (m_pColleague1)
				{
					m_pColleague1->Notify(message);
				}
			}
		}

		void SetColleague1(Colleague *pColleague)
		{
			m_pColleague1 = dynamic_cast<ConcreteColleague1 *>(pColleague);
		}

		void SetColleague2(Colleague *pColleague)
		{
			m_pColleague2 = dynamic_cast<ConcreteColleague2 *>(pColleague);
		}

	private:
		// The Mediator knows all the Colleague
		ConcreteColleague1 *m_pColleague1;
		ConcreteColleague2 *m_pColleague2;
	};

	void ConcreteColleague1::Send(wchar_t *message)
	{
		// The second parameter mark where the message comes from
		m_pMediator->Sent(message, this);
	}

	void ConcreteColleague2::Send(wchar_t *message)
	{
		m_pMediator->Sent(message, this);
	}

};
void mdiTest() {
	// Create the mediator
	mdi::Mediator *pMediator = new mdi::ConcreteMediator();

	mdi::Colleague *pColleague1 = new mdi::ConcreteColleague1(pMediator);
	mdi::Colleague *pColleague2 = new mdi::ConcreteColleague2(pMediator);

	mdi::ConcreteMediator *pConcreteMediator = dynamic_cast<mdi::ConcreteMediator *>(pMediator);
	pConcreteMediator->SetColleague1(pColleague1);
	pConcreteMediator->SetColleague2(pColleague2);

	wchar_t message[260] = L"Where are you from?";
	pColleague1->Send(message);

	return;
};

//������ģʽ
namespace itr {

	typedef struct tagNode
	{
		int value;
		tagNode *pNext;
	}Node;

	class JTList
	{
	public:
		JTList() : m_pHead(NULL), m_pTail(NULL) {};
		JTList(const JTList &);
		~JTList();
		JTList &operator=(const JTList &);

		long GetCount() const;
		Node *Get(const long index) const;
		Node *First() const;
		Node *Last() const;
		bool Includes(const int &) const;

		void Append(const int &);
		void Remove(Node *pNode);
		void RemoveAll();

	private:
		Node *m_pHead;
		Node *m_pTail;
		long m_lCount;
	};

	class Iterator
	{
	public:
		virtual void First() = 0;
		virtual void Next() = 0;
		virtual bool IsDone() const = 0;
		virtual Node *CurrentItem() const = 0;
	};

	class JTListIterator : public Iterator
	{
	public:
		JTListIterator(JTList *pList) : m_pJTList(pList), m_pCurrent(NULL) {}

		virtual void First();
		virtual void Next();
		virtual bool IsDone() const;
		virtual Node *CurrentItem() const;

	private:
		JTList *m_pJTList;
		Node *m_pCurrent;
	};

	JTList::~JTList()
	{
		Node *pCurrent = m_pHead;
		Node *pNextNode = NULL;
		while (pCurrent)
		{
			pNextNode = pCurrent->pNext;
			delete pCurrent;
			pCurrent = pNextNode;
		}
	}

	long JTList::GetCount()const
	{
		return m_lCount;
	}

	Node *JTList::Get(const long index) const
	{
		// The min index is 0, max index is count - 1
		if (index > m_lCount - 1 || index < 0)
		{
			return NULL;
		}

		int iPosTemp = 0;
		Node *pNodeTemp = m_pHead;
		while (pNodeTemp)
		{
			if (index == iPosTemp++)
			{
				return pNodeTemp;
			}
			pNodeTemp = pNodeTemp->pNext;
		}
		return NULL;
	}

	Node *JTList::First() const
	{
		return m_pHead;
	}

	Node *JTList::Last() const
	{
		return m_pTail;
	}

	bool JTList::Includes(const int &value) const
	{
		Node *pNodeTemp = m_pHead;
		while (pNodeTemp)
		{
			if (value == pNodeTemp->value)
			{
				return true;
			}
			pNodeTemp = pNodeTemp->pNext;
		}
		return false;
	}

	void JTList::Append(const int &value)
	{
		// Create the new node
		Node *pInsertNode = new Node;
		pInsertNode->value = value;
		pInsertNode->pNext = NULL;

		// This list is empty
		if (m_pHead == NULL)
		{
			m_pHead = m_pTail = pInsertNode;
		}
		else
		{
			m_pTail->pNext = pInsertNode;
			m_pTail = pInsertNode;
		}
		++m_lCount;
	}

	void JTList::Remove(Node *pNode)
	{
		if (pNode == NULL || m_pHead == NULL || m_pTail == NULL)
		{
			return;
		}

		if (pNode == m_pHead) // If the deleting node is head node
		{
			Node *pNewHead = m_pHead->pNext;
			m_pHead = pNewHead;
		}
		else
		{
			// To get the deleting node's previous node
			Node *pPreviousNode = NULL;
			Node *pCurrentNode = m_pHead;
			while (pCurrentNode)
			{
				pPreviousNode = pCurrentNode;
				pCurrentNode = pCurrentNode->pNext;
				if (pCurrentNode == pNode)
				{
					break;
				}
			}

			// To get the deleting node's next node
			Node *pNextNode = pNode->pNext;

			// If pNextNode is NULL, it means the deleting node is the tail node, we should change the m_pTail pointer
			if (pNextNode == NULL)
			{
				m_pTail = pPreviousNode;
			}

			// Relink the list
			pPreviousNode->pNext = pNextNode;
		}

		// Delete the node
		delete pNode;
		pNode = NULL;
		--m_lCount;
	}

	void JTList::RemoveAll()
	{
		delete this;
	}

	void JTListIterator::First()
	{
		m_pCurrent = m_pJTList->First();
	}

	void JTListIterator::Next()
	{
		m_pCurrent = m_pCurrent->pNext;
	}

	bool JTListIterator::IsDone() const
	{
		return m_pCurrent == m_pJTList->Last()->pNext;
	}

	Node *JTListIterator::CurrentItem() const
	{
		return m_pCurrent;
	}

};
void itrTest() {
	itr::JTList *pJTList = new itr::JTList;
	pJTList->Append(10);
	pJTList->Append(20);
	pJTList->Append(30);
	pJTList->Append(40);
	pJTList->Append(50);
	pJTList->Append(60);
	pJTList->Append(70);
	pJTList->Append(80);
	pJTList->Append(90);
	pJTList->Append(100);

	itr::Iterator *pIterator = new itr::JTListIterator(pJTList);

	// Print the list by JTListIterator
	for (pIterator->First(); !pIterator->IsDone(); pIterator->Next())
	{
		cout << pIterator->CurrentItem()->value << "->";
	}
	cout << "NULL" << endl;

	// Test for removing
	itr::Node *pDeleteNode = NULL;
	for (pIterator->First(); !pIterator->IsDone(); pIterator->Next())
	{
		pDeleteNode = pIterator->CurrentItem();
		if (pDeleteNode->value == 100)
		{
			pJTList->Remove(pDeleteNode);
			break;
		}
	}

	// Print the list by JTListIterator
	for (pIterator->First(); !pIterator->IsDone(); pIterator->Next())
	{
		cout << pIterator->CurrentItem()->value << "->";
	}
	cout << "NULL" << endl;

	delete pIterator;
	delete pJTList;

	return;
};

//������ģʽ
namespace vtr {
	class ConcreteElementA;
	class ConcreteElementB;

	class Visitor
	{
	public:
		virtual void VisitConcreteElementA(ConcreteElementA *pElementA) = 0;
		virtual void VisitConcreteElementB(ConcreteElementB *pElementB) = 0;
	};

	class ConcreteVisitor1 : public Visitor
	{
	public:
		void VisitConcreteElementA(ConcreteElementA *pElementA);
		void VisitConcreteElementB(ConcreteElementB *pElementB);
	};

	void ConcreteVisitor1::VisitConcreteElementA(ConcreteElementA *pElementA)
	{
		// ���ڸ��ݴ�������pElementA�����Զ�ConcreteElementA�е�element���в���
	}

	void ConcreteVisitor1::VisitConcreteElementB(ConcreteElementB *pElementB)
	{
		// ���ڸ��ݴ�������pElementB�����Զ�ConcreteElementB�е�element���в���
	}

	class ConcreteVisitor2 : public Visitor
	{
	public:
		void VisitConcreteElementA(ConcreteElementA *pElementA);
		void VisitConcreteElementB(ConcreteElementB *pElementB);
	};

	void ConcreteVisitor2::VisitConcreteElementA(ConcreteElementA *pElementA)
	{
		// ...
	}

	void ConcreteVisitor2::VisitConcreteElementB(ConcreteElementB *pElementB)
	{
		// ...
	}

	// Element object
	class Element
	{
	public:
		virtual void Accept(Visitor *pVisitor) = 0;
	};

	class ConcreteElementA : public Element
	{
	public:
		void Accept(Visitor *pVisitor);
	};

	void ConcreteElementA::Accept(Visitor *pVisitor)
	{
		pVisitor->VisitConcreteElementA(this);
	}

	class ConcreteElementB : public Element
	{
	public:
		void Accept(Visitor *pVisitor);
	};

	void ConcreteElementB::Accept(Visitor *pVisitor)
	{
		pVisitor->VisitConcreteElementB(this);
	}

	// ObjectStructure�࣬��ö������Ԫ�أ������ṩһ���߲�Ľӿ�����������߷�������Ԫ��
	class ObjectStructure
	{
	public:
		void Attach(Element *pElement);
		void Detach(Element *pElement);
		void Accept(Visitor *pVisitor);

	private:
		vector<Element *> elements;
	};

	void ObjectStructure::Attach(Element *pElement)
	{
		elements.push_back(pElement);
	}

	void ObjectStructure::Detach(Element *pElement)
	{
		vector<Element *>::iterator it = find(elements.begin(), elements.end(), pElement);
		if (it != elements.end())
		{
			elements.erase(it);
		}
	}

	void ObjectStructure::Accept(Visitor *pVisitor)
	{
		// Ϊÿһ��element����visitor�����ж�Ӧ�Ĳ���
		for (vector<Element *>::const_iterator it = elements.begin(); it != elements.end(); ++it)
		{
			(*it)->Accept(pVisitor);
		}
	}
};
void vtrTest() {
	vtr::ObjectStructure *pObject = new vtr::ObjectStructure;

	vtr::ConcreteElementA *pElementA = new vtr::ConcreteElementA;
	vtr::ConcreteElementB *pElementB = new vtr::ConcreteElementB;

	pObject->Attach(pElementA);
	pObject->Attach(pElementB);

	vtr::ConcreteVisitor1 *pVisitor1 = new vtr::ConcreteVisitor1;
	vtr::ConcreteVisitor2 *pVisitor2 = new vtr::ConcreteVisitor2;

	pObject->Accept(pVisitor1);
	pObject->Accept(pVisitor2);

	if (pVisitor2) delete pVisitor2;
	if (pVisitor1) delete pVisitor1;
	if (pElementB) delete pElementB;
	if (pElementA) delete pElementA;
	if (pObject) delete pObject;

	return;
};

//����¼ģʽ-----
namespace mem {

};
void memTest() {
	return;
};

//������ģʽ----
namespace itp {

};
void itpTest() {
	return;
};


int main(int argc, char *argv[])
{
	cout << "����ģʽ" << endl;
	sngTest();
	system("pause");

	cout << "ԭ��ģʽ" << endl;
	ptpTest();
	system("pause");

	cout << "��������" << endl;
	facTest();
	system("pause");

	cout << "���󹤳�" << endl;
	absfacTest();
	system("pause");

	cout << "������" << endl;
	bdrTest();
	system("pause");

	cout << "����ģʽ" << endl;
	prxTest();
	system("pause");

	cout << "������ģʽ" << endl;
	adpTest();
	system("pause");

	cout << "�Ž�ģʽ" << endl;
	bdgTest();
	system("pause");

	cout << "װ��ģʽ" << endl;
	decTest();
	system("pause");

	cout << "���ģʽ" << endl;
	fcdTest();
	system("pause");

	cout << "��Ԫģʽ" << endl;
	fywTest();
	system("pause");

	cout << "���ģʽ" << endl;
	cmpTest();
	system("pause");

	cout << "ģ�巽��" << endl;
	tmpTest();
	system("pause");

	cout << "����ģʽ" << endl;
	stgTest();
	system("pause");

	cout << "����ģʽ" << endl;
	cmdTest();
	system("pause");

	cout << "ְ����ģʽ" << endl;
	corTest();
	system("pause");

	cout << "״̬ģʽ" << endl;
	steTest();
	system("pause");

	cout << "�۲���ģʽ" << endl;
	obsTest();
	system("pause");

	cout << "�н���ģʽ" << endl;
	mdiTest();
	system("pause");

	cout << "������ģʽ" << endl;
	itrTest();
	system("pause");

	cout << "������ģʽ" << endl;
	vtrTest();
	system("pause");

	cout << "����¼ģʽ" << endl;
	memTest();
	system("pause");

	cout << "������ģʽ" << endl;
	itpTest();
	system("pause");

	return 0;
}
