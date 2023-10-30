
def read_and_handel(path):
    allline=[]
    alllable= []
    with open(path, 'r', encoding='utf-8') as f:
        i=0
        tempword = []
        templable = []
        for ann in f.readlines():
            list1 = ann.strip().split(' ')  # 去除文本中的换行符
            #print(list1)
            if len(list1)==2 :
                 tempword.append(list1[0])
                 templable.append(list1[1])
            elif len(list1)==1:
                allline.append(tempword)
                alllable.append(templable)
                tempword = []
                templable = []
            i=i+1
    #拆分训练集和测试集
    allnum = len(allline)
    splitnum = int(0.8*len(allline))
    traindata =allline[0:splitnum]
    trainlable =alllable[0:splitnum]
    devdata = allline[splitnum:allnum]
    devlable = alllable[splitnum:allnum]
    #训练
    with open('./data/train/source.txt'  , 'w', encoding='utf-8') as f:
        for t in traindata:
            f.write(' '.join(t)+'\n')
    with open('./data/train/target.txt', 'w', encoding='utf-8') as f:
        for t in trainlable:
            f.write(' '.join(t) + '\n')
    #验证
    with open('./data/dev/source.txt'  , 'w', encoding='utf-8') as f:
        for d in devdata:
            f.write(' '.join(d)+'\n')
    with open('./data/dev/target.txt', 'w', encoding='utf-8') as f:
        for d in devlable:
            f.write(' '.join(d) + '\n')

if __name__ == '__main__':
    org_path = './data/dataset.txt'
    read_and_handel(org_path)