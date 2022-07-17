%matplotlib inlline 
import math
import torch
from torch import nn
from torch.nn import functional as F
import collections
import re
from d2l import torch as d2l


d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_text():
    with open(d2l.download('time_machine'), 'r') as f:
        #按行读取文件内容
        lines = f.readlines()
    #strip()默认移除字符串开头和结尾的空白，lower()转为小写
    #re.sub()将非A-Z与a-z的符号去除,包括标点符号
    return[re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


#将文本行拆分为单词或字符标记（token）列表/词元
def tokenize(lines, token='word'):
    if token == 'word':
        #返回结果一个词作为一个token，[[],[],[]]
        return [line.split() for line in lines]
    elif token == 'char':
        #返回结果一个字符作为一个tokrn
        return [list[line] for line in lines]
    else:
        print('错误：未知令牌类型：'+token)
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
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        #未知词的处理,索引为0
        #实现索引到token的映射
        self.idx_to_token = ['<unk>'] + reserved_tokens
        #构建索引和token的字典
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)
    #查找索引
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
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

def count_corpus(tokens):  
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_text(max_tokens=-1):
    lines = read_text()
    tokens=tokenize(lines)
    vocab=Vocab(tokens)

    #将所有文本展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens>0:
        corpus = corpus[:max_tokens]
    return corpus, vocab






