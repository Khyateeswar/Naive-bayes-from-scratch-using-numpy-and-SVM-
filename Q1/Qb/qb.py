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

def part_b():
    acc=0
    for i in range(len_test_pos):
        if(random.random()>=0.5):
            acc=acc+1
    for i in range(len_test_neg):
        if(random.random()<=0.5):
            acc = acc+1
    print("random prediction accuracy = "+str(acc/(len_test_pos+len_test_neg)))
    print("predicting everything as positive accuracy = "+str(len_test_pos/(len_test_pos+len_test_neg)))
    return

part_b()