import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

epoch = 1000

sentences = [
    # eco_input                  dec_input                  dec_output
    ['ich mochte ein biter P','S i want a beer .','i want a beer . E'],
    ['ich mochte ein cola P','S i want a coke .','i want a coke . E']
]

# 建立词表
src_v = {'P':0, 'ich':1,'mochte':2, 'ein':3,'biter':4, 'cola':5}
src_id2wrd = {i:w for i,w in enumerate(src_v)}
src_vlen = len(src_v)

tgt_v = {'P':0,'i':1,'want':2, 'a':3, 'beer':4,'coke':5,'S':6,'E':7,'.':8}
idx2word = {i:w for i,w in enumerate(tgt_v)}
tg_len = len(tgt_v)

src = 5 ##eco_input最大长度
target_len = 6 # dec_input = dec_output 最大长度

d_model = 512 # 位置编码维度
d_ff = 2048 # 隐藏层单元数 , 全连接层2层的有个隐藏层
d_k = d_v = 64 # 注意力维度 k(q),v
n_layars = 6 # block数, 一个block包含多头注意力层和前馈层
n_heads = 8 # attention头数


def make_data(sentences):
    eco_input, dec_input, dec_output = [],[],[]
    for i in range(sentences):
        eco_input.extend([src_v[n] for n in sentences[i][0].split()] )
        dec_input.extend([tgt_v[w] for w in sentences[i][1].split()])
        dec_output.extend([tgt_v[w] for w in sentences[i][2].split()])

    return torch.LongTensor(eco_input), torch.LongTensor(dec_input),torch.LongTensor(dec_output)


eco_input, dec_input,dec_output = make_data(sentences)


class mySet(data.Dataset):
    def __init__(self, eco_input, dec_input,dec_output):
        super().__init__()
        self.eco_input = eco_input
        self.dec_input = dec_input
        self.dec_output = dec_output
    def __len__(self):
        return self.eco_input.shap[0]
    def __getitem__(self, index) -> Any:
        return self.eco_input[index], self.dec_input[index],self.dec_output[index]


loader = data.DataLoader(mySet(eco_input,dec_input,dec_output), 2,True)


#Transformer

class PositionalEncoding(nn.Module):
    """
    正弦位置编码，即通过三角函数构建位置编码

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    """

    def __init__(self, dim: int, dropout=0.1, max_len=5000):
        """
        :param dim: 位置向量的向量维度，一般与词向量维度相同，即d_model
        :param dropout: Dropout层的比率
        :param max_len: 句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 位置编码的Dropout层
        


        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # 初始化pe
        position = torch.arange(0, max_len).unsqueeze(1)  # 构建pos，为句子的长度，相当于pos
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * torch.tensor(
            -(math.log(10000.0) / dim))))  # 复现位置编码sin/cos中的公式
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # 偶数使用sin函数
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # 奇数使用cos函数
        pe = pe.unsqueeze(1)  # 扁平化成一维向量

        self.register_buffer('pe', pe)  # pe不是模型的一个参数，通过register_buffer把pe写入内存缓冲区，当做一个内存中的常量
        self.dim = dim

    def forward(self, emb, step=None):
        """
        词向量和位置编码拼接并输出
        :param emb: 词向量序列（FloatTensor），``(seq_len, batch_size, self.dim)``
        :param step: 如果 stepwise("seq_len=1")，则用此位置的编码
        :return: 词向量和位置编码的拼接
        """
        emb = emb + self.pe[:emb.size(0),:]  # 位置编码
        return self.dropout(emb)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(src_vlen,d_model) # token 单词编码
        self.pos_encoder = PositionalEncoding(d_model) # 位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layars)])  #将block放在一起
    
    def forwarforward(self, enc_input):
        enc_out = self.embed(enc_input) # [batch,seq_len,d_model] 编码维度512
        enc_out = self.pos_encoder([enc_out.transpose(0,1)]).transpose(0,1)
        enc_atttion_mask = get_attn_pad_mask(enc_input, enc_input)
        enc_atttions = []
        for layer in self.layers:
            enc_out, enc_atttion = layer(enc_out, enc_atttion_mask)
            enc_atttions.append(enc_atttion)
        return enc_out, enc_atttions




class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.project = nn.Linear(d_model, tgt_vlen,cias = False)
    
    def forward(self,enc_input,dec_input):
        enc_output,enc_attention = self.encoder(enc_input)
        dec_output,dec_attention,dec_enc_atttention = self.decoder(dec_input,enc_input,enc_output)
        dec_pre = self.project(dec_output)
        return dec_pre.view(-1,dec_pre.size(-1)),enc_attention,dec_attention,dec_enc_atttention
    

model = Transformer()
