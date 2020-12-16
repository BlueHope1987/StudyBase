[(i,'*',j,'=',i*j) for i in range(10) for j in range(10)]
while True:
    line = raw_input('show me the money>')
    if line =='five dollar':
        break
    print line
print 'Thank you!'