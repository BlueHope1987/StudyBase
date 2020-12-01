#https://github.com/microsoft/ai-edu/blob/master/B-%E6%95%99%E5%AD%A6%E6%A1%88%E4%BE%8B%E4%B8%8E%E5%AE%9E%E8%B7%B5/B13-AI%E5%AF%B9%E8%81%94%E7%94%9F%E6%88%90%E6%A1%88%E4%BE%8B/README.md
#分隔数据的python代码 (split_data.py)

import sys

filename = sys.argv[1]
with open(filename, 'r', encoding='utf-8') as infile:
    with open(filename + '.clean', 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        for line in lines:
            out = ""
            for i in line.strip():
                out += i + (' ')
            out = out[:-1]
            out += '\n'
            outfile.write(out)

#执行如下命令完成文件分隔
#python split_data.py train.txt.up
#python split_data.py train.txt.down