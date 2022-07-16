import os
import torch
from d2l import torch as d2l

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip','94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data():
    data_dir=d2l.download_extract('fra-eng')
    with open (os.path.join(data_dir, 'fra.txt'), 'f', encoding='utf-8') as f:
        return f.read()

raw_text=read_data()
print(raw_text[:75])
#预处理
def preprocess_data(text):
    def no_space(char, prev_char):
        return char in set(',.!?' ) and prev_char != ' '
    #替换不间断空格
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' '+ char if i>0 and no_space(char, text[i-1]) else char for i , char in enumerate(text)]
    return ''.join(out)

text = preprocess_data(raw_text)
