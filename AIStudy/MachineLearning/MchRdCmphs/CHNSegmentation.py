#中文分词 基于匹配的分词
#逆向最大匹配算法 从后往前匹配最长的词
#P32

'''
逆向最大匹配算法
输入语句s和词表vocab，输出分词列表。
例子：
输入：s='今天天气真不错'，vocab=['天气','今天','昨天','真','不错','真实','天天']
输出：['今天','天气','真','不错']
'''

def backward_maximal_matching(s,vocab):
    result=[]
    end_pos=len(s)
    while end_pos>0:
        found=False
        for start_pos in range(end_pos): # (int) in range((int))
            if s[start_pos:end_pos] in vocab: # if s[(int):(int)] in vocab s[0:7] s[1:7] s[2:7]...
                #找到最长匹配的单词，放在分词结果最前面
                result=[s[start_pos:end_pos]]+result
                found=True
                break
        if found:
            end_pos=start_pos
        else:
                #未找到匹配的单词，将单字作为词分出
                result=[s[end_pos-1]]+result
                end_pos-=1
    return result

print(backward_maximal_matching('今天天气真不错',['天气','今天','昨天','真','不错','真实','天天']))

#中文分词可用jieba软件包实现
#pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
import jieba
seg_list=jieba.cut('我来到北京清华大学')
print('/'.join(seg_list))

#英文分词可用spaCy软件包实现
#pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple
#python -m spacy download en_core_web_sm
import spacy
nlp=spacy.load('en_core_web_sm')
text=('Today is very special. I just got my Ph.D. degree.')
doc=nlp(text)
print({e.text for e in doc})


#P47 词性标注

#中文命名实体识别和词性标注可以用jieba软件包实现
import jieba.posseg as pseg
words=pseg.cut("我爱北京天安门")
for word, pos in words:
    print('%s %s'%(word,pos))

#英文命名实体识别和词性标注可以通过spaCy软件包实现
import spacy
nlp=spacy.load('en_core_web_sm')
doc=nlp(u"Apple may buy a U.K. startup for $1 billion")
print('---------------Part of Speech-----------')
for token in doc:
    print(token.text,token.pos_)
print('---------Named Entity Recognition--------')
for ent in doc.ents:
    print(ent.text,ent.label_)
