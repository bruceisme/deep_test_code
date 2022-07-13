import collections
from operator import index
import re
from d2l import torch as d2l

def read_text():
    with open('文件路径', 'r') as f:
        #按行读取文件内容
        lines = f.readlines()
    #strip()默认移除字符串开头和结尾的空白，lower()转为小写
    #re.sub()将非A-Z与a-z的符号去除,包括标点符号
    return[re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_text()

#将文本行拆分为单词或字符标记（token）列表/词元
def tokenize(lines, token='word'):
    if token == 'word':
        #返回结果一个词作为一个token
        return [line.split() for line in lines]
    elif token == 'char':
        #返回结果一个字符作为一个tokrn
        return [list[line] for line in lines]
    else:
        print('错误：未知令牌类型：'+token)

def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0], list):
        #先循环tokens里的line，再循环line里的token
        tokens=[token for line in tokens for token in line ]
        return collections.Counter(tokens)

class Vocab:
    #将token映射到从0开始的数字索引中
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens=[]
        if reserved_tokens is None:
            reserved_tokens=[]
        #统计出现频率，并排序
        counter = count_corpus(tokens)
        #根据频率进行降序排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        #未知词的处理,索引为0
        #实现索引到token的映射
        self.idx_to_token = ['<UNK>']+reserved_tokens
        #构建索引和token的字典
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token) }

        for token, freq in self._token_freqs:
            #如果一个词的频率小于某个值,丢弃
            if freq<min_freq:
                break
            if token not in self.token_to_idx:
                #添加至idx_to_token和token_to_idx中
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
    def __len__(self):
        return len(self.idx_to_token)
    
    #查找索引
    def __gititem__(self, tokens):
        #先判断是否是列表或者元组（是否只是一个token）
        if not isinstance(tokens, (list, tuple)):
            #如果不是2D
            #找不到，输出默认值self.unk
            return self.token_to_idx.get(tokens, self.unk)
        return[self.__gititem__(token) for token in tokens]
    
    #查找字符
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0
    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_text(max_tokens=-1):
    lines = read_text()
    tokens=tokenize(lines)
    vocab=Vocab(tokens)

    #将所有文本展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens>0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab=load_corpus_text()




