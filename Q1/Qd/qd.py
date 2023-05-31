import glob
import sys
import math
import random
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lem = WordNetLemmatizer()



sw_nltk_list = stopwords.words('english') # stopwords to remove them before stemming
waste = ['<','>','/','.','br','(',')',',','<br>',"/>",'><','/><br>','\'','/><br']
sw_nltk_list.extend(waste)
sw_nltk = {}
for i in range(len(sw_nltk_list)):
    sw_nltk[sw_nltk_list[i]]=1
    

print(sw_nltk)
ps = PorterStemmer()# stemming function



train_path = str(sys.argv[1])
test_path = str(sys.argv[1])
# train_path = 'part1_data/train'
# test_path = 'part1_data/test'

train_pos_files = glob.glob(train_path+'/pos/*.txt')
train_neg_files = glob.glob(train_path+'/neg/*.txt')

test_pos_files = glob.glob(test_path+'/pos/*.txt')
test_neg_files = glob.glob(test_path+'/neg/*.txt')

len_train_pos = len(train_pos_files)
len_train_neg = len(train_neg_files)

len_test_pos = len(test_pos_files)
len_test_neg = len(test_neg_files)


# print(len_train_pos)
# print(len_train_neg)

# print(len_test_pos)
# print(len_test_neg)
def predict_stem(s,pos_voc,neg_voc,len_pos_tot,len_neg_tot):
    s=s.split()
    p1=0
    n1=0
    for i in range(len(s)):
        s[i] = ps.stem(s[i])
        if not (sw_nltk.get(s[i])!=None):
            if pos_voc.get(s[i])!=None or neg_voc.get(s[i])!=None :
                if pos_voc.get(s[i])!=None:
                    p1 = p1 + math.log((pos_voc.get(s[i])+1)/(len_pos_tot+len(pos_voc)))
                else:
                    p1 = p1 + math.log(1/(len(pos_voc)+len_pos_tot))
                if(neg_voc.get(s[i])!=None):
                    n1 = n1 + math.log((neg_voc.get(s[i])+1)/(len_neg_tot+len(neg_voc)))
                else:
                    n1 = n1 + math.log(1/(len(neg_voc)+len_neg_tot))
    p1 = p1+math.log(len_train_pos/(len_train_pos+len_train_neg)) #prior probability
    n1 = n1 + math.log(len_train_neg/(len_train_pos+len_train_neg))
    if(p1>=n1):
        return True
    return False


def show_pos_wc(sort_pos_voc):
    s={}
    for i in range(2000):
        s[sort_pos_voc[i][0]]=sort_pos_voc[i][1]      
    stopwords = set(STOPWORDS)
    pos_wc = WordCloud(background_color='white',max_words=2000,stopwords=stopwords)
    pos_wc.generate_from_frequencies(s)
    plt.imshow(pos_wc, interpolation='bilinear')
    plt.axis('off')
    plt.show(block=False)
    plt.savefig("pos_wordcloud.png")
    return

def show_neg_wc(sort_neg_voc):
    s={}
    for i in range(2000):
        s[sort_neg_voc[i][0]]=sort_neg_voc[i][1]
    stopwords = set(STOPWORDS)
    neg_wc = WordCloud(background_color='white',max_words=2000,stopwords=stopwords)
    neg_wc.generate_from_frequencies(s)
    plt.imshow(neg_wc, interpolation='bilinear')
    plt.axis('off')
    plt.show(block=False)
    plt.savefig("neg_wordcloud.png")
    return

def part_d():
    pos_voc={}
    neg_voc={}
    len_pos_tot =0
    len_neg_tot = 0
    for i in range(len_train_pos):
        f = open(train_pos_files[i], "r",encoding='utf8')
        s=f.read()
        s=s.split()
        for j in range(len(s)):
            if not ( sw_nltk.get(s[j])!=None):
                len_pos_tot=len_pos_tot+1
                s[j]=ps.stem(s[j])
                if pos_voc.get(s[j])!=None:
                    pos_voc[s[j]] = pos_voc[s[j]]+1
                else:
                    pos_voc[s[j]]=1

                    
    for i in range(len_train_neg):
        f = open(train_neg_files[i], "r",encoding='utf8')
        s=f.read()
        s=s.split()
        for j in range(len(s)):
            if not (sw_nltk.get(s[j])!=None):
                len_neg_tot=len_neg_tot+1
                s[j]=ps.stem(s[j])
                if neg_voc.get(s[j])!=None:
                    neg_voc[s[j]] = neg_voc[s[j]]+1
                else:
                    neg_voc[s[j]]=1
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len_train_pos):
        f = open(train_pos_files[i], "r",encoding='utf8')
        s=f.read()
        if (predict_stem(s,pos_voc,neg_voc,len_pos_tot,len_neg_tot)):
            tp = tp+1 
        else:
            fn = fn+1
        
    for i in range(len_train_neg):
        f = open(train_neg_files[i], "r",encoding='utf8')
        s=f.read()
        if (predict_stem(s,pos_voc,neg_voc,len_pos_tot,len_neg_tot)):
            fp = fp+1 
        else:
            tn = tn+1
    print(tp)
    print(fp)
    print(fn)
    print(tn)
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1_score = 2*(precision*recall)/(precision+recall)
    print(accuracy)
    print(precision)
    print(recall)
    print(F1_score)
    
    
    
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len_test_pos):
        f = open(test_pos_files[i], "r",encoding='utf8')
        s=f.read()
        if (predict_stem(s,pos_voc,neg_voc,len_pos_tot,len_neg_tot)):
            tp = tp+1 
        else:
            fn = fn+1
        
    for i in range(len_test_neg):
        f = open(test_neg_files[i], "r",encoding='utf8')
        s=f.read()
        if (predict_stem(s,pos_voc,neg_voc,len_pos_tot,len_neg_tot)):
            fp = fp+1 
        else:
            tn = tn+1
    print(tp)
    print(fp)
    print(fn)
    print(tn)
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1_score = 2*(precision*recall)/(precision+recall)
    print(accuracy)
    print(precision)
    print(recall)
    print(F1_score)
    srt_pos_voc = sorted(pos_voc.items(),key=lambda kv:(kv[1], kv[0]),reverse = True)
    srt_neg_voc = sorted(neg_voc.items(),key=lambda kv:(kv[1], kv[0]),reverse = True)
    show_pos_wc(srt_pos_voc)
    show_neg_wc(srt_neg_voc)
    return
        


part_d()