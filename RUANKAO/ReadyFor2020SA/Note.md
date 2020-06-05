软考 备战2020下半年 系统架构设计师


20200604 每日一练 80
https://uc.educity.cn/tiku/testReport.html?id=7291378

    分布式系统开发分为五个逻辑计算层：表示层实现用户界面；表示逻辑层为了生成数据表示而必须进行的处理任务，如输入数据编辑等；应用逻辑层包括为支持实际业务应用和规则所需的应用逻辑和处理过程，如信用检查、数据计算和分析等；数据处理层包括存储和访问数据库中的数据所需的应用逻辑和命令，如查询语句和存储过程等；数据层是数据库中实际存储的业务数据。

    大型局域网的层次和各个层次的功能，大型局域网通常划分为核心层、汇聚层和接入层，其中核心层在逻辑上只有一个，它连接多个分布层交换机，通常是一个园区中连接多个建筑物的总交换机的核心网络设备；汇聚层定义的网络的访问策略；接入层提供局域网络接入功能，可以使用集线器代替交换机。

    数学建模是利用数学方法解决实际问题的一种实践。即通过抽象、简化、假设、引进变量等处理过程后，将实际问题用数学方式表达，建立起数学模型，然后运用先进的数学方法及计算机技术进行求解。 这是A和B的原因，数学模型是对于现实世界的一个特定对象，一个特定目的，根据特有的内在规律，做出一些必要的假设，运用适当的数学工具，得到一个数学结构。对不同的问题，有不同的评价标准，数学模型难有统一的普适标准来评价。一般会采用相同的方法来评价数学建模（从这个角度来看是有统一的、普适性的标准），但是对于大部分方法中，还是包含主观评价和客观评价2类步骤的。所以从评分标准这个角度而言，一般是没有统一的、普适的标准的。在数学建模过程中是需要从失败和用户反馈中学习和改进。

系统测试将软件、硬件、网络等其它因素结合，对整个软件进行测试。（  ）不是系统测试的内容。
A.路径测试    B.可靠性测试    C.安装测试    D.安全测试
R:A W:C N:软件测试

    系统测试是将已经确认的软件、计算机硬件、外设和网络等其他因素结合在一起，进行信息系统的各种集成测试和确认测试，其目的是通过与系统的需求相比较，发现所开发的系统与用户需求不符或矛盾的地方。系统测试是根据系统方案说明书来设计测试用例，常见的系统测试主要有恢复测试、安全性测试、压力测试、性能测试、可靠性测试、可用性测试、可维护性测试和安装测试。
    路径测试法是白盒测试，白盒测试是属于单元测试的内容。

某企业通过一台路由器上联总部，下联4个分支结构，设计人员分配给下级机构一个连续的地址空间，采用一个子网或者超网段表示。这样做的主要作用是（ ）。
A.层次化路由选择    B.易于管理和性能优化    C.基于故障排查   D.使用较少的资源
R:A W:B N:网络规划与设计

    层次化路由的含义是指对网络拓扑结构和配置的了解是局部的，一台路由器不需要知道所有的路由信息，只需要了解其管辖的路由信息，层次化路由选择需要配合层次化的地址编码。而子网或超网就属于层次化地址编码行为。

某软件公司正在设计一个图像处理软件，该软件需要支持用户在图像处理过程中的撤销和重做等动作，为了实现该功能，采用（命令模式）最为合适。

    本题主要考查对设计模式的理解和掌握。根据题干描述，系统需要支持用户在图像处理过程中的撤销和重做的动作，因此可以将用户动作封装成对象，通过对象之间的传递和转换实现撤销和重做等动作。根据上述分析，选项中列举的设计模式中，命令模式最符合要求。

某企业欲对内部的数据库进行数据集成。如果集成系统的业务逻辑较为简单，仅使用数据库中的单表数据即可实现业务功能，这时采用（主动记录）方式进行数据交换与处理较为合适；如果集成系统的业务逻辑较为复杂，并需要通过数据库中不同表的连接操作获取数据才能实现业务功能，这时采用（数据映射）方式进行数据交换与处理较为合适。

    本题主要考查数据集成的相关知识。关键要判断在进行集成时，需要数据库的单表还是多表进行数据整合。如果是单表即可完成整合，则可以将该表包装为记录，采用主动记录的方式进行集成；如果需要多张表进行数据整合，则需要采用数据映射的方式完成数据集成与处理。
    网上以及资料上，都没有标准说法。这个题中的内容本就是不标准的。
    1、包装器：由于所集成的各数据源的异构性,即存在着数据库管理系统(DBMS)的差异,或者操作系统(OS)的差异,需要对参与集成的各个数据源进行包装格式转换,向外提供统一的无差别的调用接口,与数据源相关的数据格式包装转换都是在包装器中实现,数据源通过包装器向外提供外观视图,集成器分解后基于单源的查询也是交给包装器执行,并按照集成器能理解的格式返回结果。这个类似于适配器转换。
    2、数据网关：数据网关的作用就是连接不同的通信系统，实现数据互通，所以首先要对不同系统进行物理连接，在物理连接的基础上，再根据不同系统的通信协议，利用协议允许的接口进行软件连接，通过数据接收、提取、发送的过程实现数据转发。
    3、数据映射：给定两个数据模型，在模型之间建立起数据元素的对应关系，将这一过程称为数据映射。
    4、主动记录：没有找到相应说法。从题干的描述来看，可以理解为记录到业务系统时，同步记录到对应的集成系统中的一种实时数据同步方式。因为它不需要做数据映射，直接使用该单表数据即可。
    TIP.在软考中总会碰到少量的新名词问题的，一方面可以通过练习做题来丰富，一方面学会在临考时的临场发挥。不需要因为过于在意这类问题而增加自己的心里负担。

20200605 每日一练 66
https://uc.educity.cn/tiku/testReport.html?id=7302941

系统应用架构设计中，网络架构数据流图的主要作用是将处理器和设备分配到网络中。（  ）不属于网络架构数据流图的内容。
A.服务器、客户端及其物理位置   B.处理器说明信息   C.单位时间的数据流大小   D.传输协议
R:C W:B N:网络规划与设计

    本题考查网络规划与设计的基本知识。
	应用架构建模中要绘制的第一个物理数据流图（PDFD）是网络架构DFD，它们不显示单位时间的数据流量，需要显示的信息包括服务器及其物理位置；客户端及其物理位置；处理器说明；传输协议。

某公司拟开发了个轿车巡航定速系统，系统需要持续测量车辆当前的实时速度，并根据设定的期望速度启动控制轿车的油门和刹车。针对上述需求，采用（  ）架构风格最为合适。
A.解释器   B.过程控制   C.分层  D.管道-过滤器
R:B W:A N:软件架构风格

    过程控制又称闭环风格，该风格的最大特点是设定参数，并不断测量现有的实际数据，将实际值与设定值进行比较，以确定接下来的操作。在本题中，定速巡航的场景正好符合这个模式。

    用于管理信息系统规划的方法很多，主要是关键成功因素法（Critical Success Factors，CSF）、战略目标集转化法（Strategy Set Transformation, SST）和企业系统规划法（Business System Planning, BSP）。其它还有企业信息分析与集成技术（BIAIT）、产出／方法分析（E/MA）、投资回收法（ROI）、征费法（chargout）、零线预算法、阶石法等。用得最多的是前面三种。
    1. 关键成功因素法（CSF）
    在现行系统中，总存在着多个变量影响系统目标的实现，其中若干个因素是关键的和主要的（即关键成功因素）。通过对关键成功因素的识别，找出实现目标所需的关键信息集合，从而确定系统开发的优先次序。
    关键成功因素来自于组织的目标，通过组织的目标分解和关键成功因素识别、性能指标识别，一直到产生数据字典。
    识别关键成功因素，就是要识别联系于组织目标的主要数据类型及其关系。不同的组织的关键成功因素不同，不同时期关键成功因素也不相同。当在一个时期内的关键成功因素解决后，新的识别关键成功因素又开始。
    关键成功因素法能抓住主要矛盾，使目标的识别突出重点。由于经理们比较熟悉这种方法，使用这种方法所确定的目标，因而经理们乐于努力去实现。该方法最有利于确定企业的管理目标。
    2.战略目标集转化法（SST）
    把整个战略目标看成是一个“信息集合”，由使命、目标、战略等组成，管理信息系统的规划过程即是把组织的战略目标转变成为管理信息系统的战略目标的过程。
    战略目标集转化法从另一个角度识别管理目标，它反映了各种人的要求，而且给出了按这种要求的分层，然后转化为信息系统目标的结构化方法。它能保证目标比较全面，疏漏较少，但它在突出重点方面不如关键成功因素法。
    3. 企业系统规划法（BSP）
    信息支持企业运行。通过自上而下地识别系统目标、企业过程和数据，然后对数据进行分析，自下而上地设计信息系统。该管理信息系统支持企业目标的实现，表达所有管理层次的要求，向企业提供一致性信息，对组织机构的变动具有适应性。
    企业系统规划法虽然也首先强调目标，但它没有明显的目标导引过程。它通过识别企业“过程”引出了系统目标，企业目标到系统目标的转化是通过企业过程/数据类等矩阵的分析得到的。

    架构描述语言（Architecture Description Language，ADL）是一种为明确说明软件系统的概念架构和对这些概念架构建模提供功能的语言。ADL主要包括以下组成部分：组件、组件接口、连接件和架构配置。ADL对连接件的重视成为区分ADL和其它建模语言的重要特征之一。

    架构复审一词来自于ABSD。在ABSD中，架构设计、文档化和复审是一个迭代过程。从这个方面来说，在一个主版本的软件架构分析之后，要安排一次由外部人员（用户代表和领域专家）参加的复审。
	复审的目的是标识潜在的风险，及早发现架构设计中的缺陷和错误，包括架构能否满足需求、质量需求是否在设计中得到体现、层次是否清晰、构件的划分是否合理、文档表达是否明确、构件的设计是否满足功能与性能的要求等等。
	由外部人员进行复审的目的是保证架构的设计能够公正地进行检验，使组织的管理者能够决定正式实现架构。

The architecture design specifies the overall architecture and the placement of software and hardware that will be used. Architecture design is a very complex process that is often left to experienced architecture designers and consultants. The first step is to refine the (  ) into more detailed requirements that are then employed to help select the architecture to be used and the software components to be placed on each device. In a (  ), one also has to decide whether to use a two-tier, three-tier,or n-tier architecture. Then the requirements and the architecture design are used to develop the hardware and software specification. There are four primary types of nonfunctional requirements that can be important in designing the architecture. (  ) specify the operating environment(s) in which the system must perform and how those may change over time. (   ) focus on the nonfunctional requirements issues such as response time,capacity,and reliability. (  ) are the abilities to protect the information system from disruption and data loss, whether caused by an intentional act. Cultural and political requirements are specific to the countries in which the system will be used.
A.functional requirements  B. nonfunctional requirements  C. system constraint  D. system operational environment
A. client-based architecture  B. server-based architecture  C. network architecture  D. client-server architecture
A. Operational requirements  B. Speed requirement  C. Access control requirements  D. Customization requirements
A. Environment requirements  B. Maintainability requirements  C. Performance requirements   D. Virus control requirements
A. Safety requirements   B. Security requirements   C. Data management requirements    D. System requirements
R:DBACB W:DDAAB N:专业英语

    架构设计指定了将要使用的软件和硬件的总体架构和布局。 架构设计是一个非常复杂的过程，往往留给经验丰富的架构设计师和顾问。 第一步是将（71）细化为更详细的要求，然后用于帮助选择要使用的体系结构以及要放置在每个设备上的软件组件。
    在（72）中，还必须决定是使用两层，三层还是n层架构。 然后使用需求和体系结构设计来开发硬件和软件规范。 有四种主要的非功能需求类型可能在设计架构时非常重要。 （73）指定系统必须执行的操作环境以及这些操作环境如何随时间变化。 （74）侧重于非功能性需求问题，如响应时间，容量和可靠性。 （75）是否有能力保护信息系统免受故意行为造成的破坏和数据丢失。 文化和政治要求是特定于系统将被使用的国家。
    71 A functional requirements（功能需求） B nonfunctional requirements （非功能需求）
    C system constraint （系统约束） D system operational environment （系统操作环境）
    72 A client-based architecture （基于客户端的架构）
    B server-based architecture（基于服务器的架构）
    C network architecture （网络架构）
    D client-server architecture （客户端 - 服务器架构）
    73 A operational requirements （操作要求）
    B speed requirements （速度要求）
    C Access control requirements （访问控制要求）
    D customization requirements （用户要求）
    74 A environment requirements （环境要求）
    B Maintainability requirements （可维护性要求）
    C performance requirements （性能要求）
    D virus control requirements（病毒控制要求）
    75 A safety requirements （安全要求）
    B security requirements（安全要求）
    C Data management requirements （数据管理要求）
    D system requirements（系统要求）

面向服务系统构建过程中，（SOAP）用于实现Web服务的远程调用，（BPEL）用来将分散的、功能单一的Web服务组织成一个复杂的有机应用。

    UDDI（Universal Description，Discovery＆Integration），UDDI用于Web服务注册和服务查找；
    WSDL（Web Service Description Language），WSDL用于描述Web服务的接口和操作功能；
    SOAP（Simple Object Access Protocol），SOAP为建立Web服务和服务请求之间的通信提供支持。
    BPEL（Business Process Execution Language For Web Services）翻译成中文的意思是面向Web 服务的业务流程执行语言，也有的文献简写成BPEL4WS，它是一种使用 Web 服务定义和执行业务流程的语言。使用BPEL，用户可以通过组合、编排和协调 Web 服务自上而下地实现面向服务的体系结构（SOA）。BPEL 提供了一种相对简单易懂的方法，可将多个 Web 服务组合到一个新的复合服务（称作业务流程）中。

软件架构贯穿于软件的整个生命周期，但在不同阶段对软件架构的关注力度并不相同，在（  ）阶段，对软件架构的关注最多。
A.需求分析与设计    B.设计与实现    C.实现与测试    D.部署与变更
R:B W:A N:软件架构的概念

    本题主要考查软件架构对软件开发的影响和在生命周期中的关注力度。
	软件架构贯穿于软件的整个生命周期，但在不同的阶段对软件架构的关注力度并不相同。其中需求分析阶段主要关注问题域；设计阶段主要将需求转换为软件架构模型；软件实现阶段主要关注将架构设计转换为实际的代码；软件部署阶段主要通过组装软件组件提高系统的实现效率。其中设计与实现阶段在软件架构上的工作最多，也最重要，因此关注力度最大。

    用户界面设计的3条黄金规则为：
    1、让用户拥有控制权；
    2、减少用户的记忆负担；
    3、保持界面一致。

活动定义是项目时间管理中的过程之一，（工作分解结构（WBS））是进行活动定义时通常使用的一种工具。

    活动定义的常用工具包括：
    1．分解
    采用分解技术来定义活动，就是要把项目工作包分解成更小的、更易于管理的组成部分，即活动——为完成工作包而必须开展的工作。定义活动过程最终输出的是活动，而非可交付成果。可交付成果是创建工作分解结构过程的输出。
    WBS、WBS 词典与活动清单，既可依次编制，也可同时编制。WBS和WBS 词典是制定最终活动清单的依据。WBS 中的每个工作包都需分解成活动，以便通过这些活动来完成相应的可交付成果。让团队成员参与分解，有助于得到更好、更准确的结果。
    2．滚动式规划
    滚动式规划是一种渐进明细的规划方式，即对近期要完成的工作进行详细规划，而对远期工作则暂时只在WBS 的较高层次上进行粗略规划。因此，在项目生命周期的不同阶段，工作分解的详细程度会有所不同。例如，在早期的战略规划阶段，信息尚不够明确，工作包也许只能分解到里程碑的水平；而后，随着了解到更多的信息，近期即将实施的工作包就可以分解成具体的活动。
    3．模板
    标准活动清单或以往项目的部分活动清单，经常可用做新项目的模板。模板中的活动属性信息，也有助于定义活动。模板还可用来识别典型的进度里程碑。
    4．专家判断
    富有经验并擅长制定详细项目范围说明书、工作分解结构和项目进度计划的项目团队成员或其他专家，可以为定义活动提供专业知识。