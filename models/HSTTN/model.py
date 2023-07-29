#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import math


def feature_fusion(fusion_type, time_outputs, turb_outputs, shape, fusion_conv):
    B,turb,time,dim = shape
    if fusion_type==1: #global fusion
        time_fusion = time_outputs.reshape(B, time, turb, dim).permute(0, 2, 1, 3)
        turb_fusion = turb_outputs.reshape(B, turb, time, dim)
        fusion = fusion_conv(torch.cat((time_fusion, turb_fusion), -1).permute(0, 3, 1, 2))
        #fusion = bn(fusion)
        fusion = F.relu(fusion).permute(0, 2, 3, 1)  # [B,turb,time,conv_dim]
        time_outputs = fusion.permute(0, 2, 1, 3).reshape(B, time, -1)
        turb_outputs = fusion.reshape(B, turb, -1)
        return time_outputs, turb_outputs
    elif fusion_type==2: #local fusion
        time_fusion = time_outputs.reshape(B, turb, time, dim)
        turb_fusion = turb_outputs.reshape(B, time, turb, dim).permute(0, 2, 1, 3)
        fusion = fusion_conv(torch.cat((time_fusion,turb_fusion),-1).permute(0, 3, 1, 2))
        #fusion = fusion_conv((time_fusion+turb_fusion).permute(0, 3, 1, 2))
        #fusion = bn(fusion)
        #残差学习
        #fusion = fusion + time_fusion.permute(0,3,1,2) + turb_fusion.permute(0,3,1,2)
        fusion = F.relu(fusion).permute(0, 2, 3, 1)  #[B, turb, time, conv_dim]
        time_out = fusion.reshape(-1, time, dim)
        turb_out = fusion.permute(0, 2, 1, 3).reshape(-1, turb, dim)
        return time_out , turb_out

def up_conv(time_dec_outputs, time_enc_outputs, shape, upconv ):
    B, turb, time, dim = shape
    dec_outputs = time_dec_outputs.reshape(B, turb, time, dim).permute(0, 3, 1, 2)
    enc_outputs = time_enc_outputs.reshape(B, turb, time, dim).permute(0, 3, 1, 2)
    outputs = torch.cat((enc_outputs, dec_outputs),1)
    #outputs = enc_outputs + dec_outputs
    outputs = upconv(outputs)
    #outputs = bn(outputs)
    outputs = F.relu (outputs).permute(0, 2, 3, 1)
    time_outputs = outputs.reshape(B*turb, -1, dim)
    turb_outputs = outputs.permute(0, 2, 1, 3).reshape(-1, turb, dim)
    return time_outputs, turb_outputs


# ====================================================================================================
# Transformer模型

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)


def get_attn_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self,
            d_k,
        ):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        #print("Q shape:{}, K shape:{}, V shape:{}".format(Q.shape,K.shape,V.shape))
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask.bool(), -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """

    def __init__(self,
            Q_d_model, K_d_model, d_k, d_v, n_heads
        ):
        super(MultiHeadAttention, self).__init__()
        self.Q_d_model, self.d_k, self.d_v, self.n_heads = Q_d_model, d_k, d_v, n_heads
        self.W_Q = nn.Linear(Q_d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(K_d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(K_d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, Q_d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        #print("input Q shape:{}, K shape:{}, V shape:{}".format(input_Q.shape,input_K.shape,input_V.shape))
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        #print("input Q shape:{}, input K shape:{}, input V shape:{}".format(input_Q.shape,input_K.shape,input_V.shape))
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 再做一个projection
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.Q_d_model).to(input_Q.device)(output + residual), attn


# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,
            d_model,d_ff
        ):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(inputs.device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self,
            d_model, d_k, d_v, n_heads, d_ff
        ):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_model,d_k,d_v,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self,
            selfattn_d_model,
            decenc_d_model,
            d_k,d_v,n_heads,d_ff
        ):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(selfattn_d_model,selfattn_d_model,d_k,d_v,n_heads)
        self.dec_enc_attn = MultiHeadAttention(selfattn_d_model,decenc_d_model,d_k,d_v,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(selfattn_d_model,d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)  # 这里的Q,K,V全是Decoder自己的输入
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        # zy
        #dec_outputs = dec_inputs
        dec_self_attn = []
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


class Encoder(nn.Module):
    def __init__(self,
        config,
        input_dim,
        fusion_type=2, # 1:token=time/turb * F 2:token = F
        conv_dim = 32,
        d_model = 512,
        d_k = 64,
        d_v = 64,
        n_enc_layers = 6,
        n_heads = 8,
        d_ff = 1024
        ):
        super(Encoder, self).__init__()
        self.fusion_type=fusion_type
        self.n_enc_layers = n_enc_layers
        if fusion_type==1:
            time_enc_dmodel = config['Turbins']*conv_dim
            turb_enc_dmodel = config['in_len']*conv_dim
            self.time_pos_emb = PositionalEncoding(config['Turbins']*conv_dim)  # Transformer中位置编码时固定的，不需要学习
            self.turb_pos_emb = PositionalEncoding(config['in_len']*conv_dim)
        elif fusion_type==2:
            time_enc_dmodel = conv_dim
            turb_enc_dmodel = conv_dim
            self.time_pos_emb = PositionalEncoding(conv_dim)  # Transformer中位置编码时固定的，不需要学习
            self.turb_pos_emb = PositionalEncoding(conv_dim)

        self.conv = nn.Conv2d(in_channels=input_dim,out_channels=conv_dim,kernel_size=1)

        self.fusion_conv_1 = nn.Conv2d(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=1)
        self.fusion_conv_3 = nn.Conv2d(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=1)
        self.fusion_conv_6 = nn.Conv2d(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=1)

        self.maxpool_3 = nn.AvgPool2d(kernel_size=(1,3),stride=(1,3))
        self.maxpool_6 = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        # 神经网络参数初始化
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        '''
        self.time_enc_layers_1 = nn.ModuleList(
            [EncoderLayer(time_enc_dmodel,d_k,d_v,n_heads,d_ff) for _ in range(n_enc_layers)])
        self.turb_enc_layers_1 = nn.ModuleList(
            [EncoderLayer(turb_enc_dmodel,d_k,d_v,n_heads,d_ff) for _ in range(n_enc_layers)])
        self.time_enc_layers_3 = nn.ModuleList(
            [EncoderLayer(time_enc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_enc_layers)])
        self.turb_enc_layers_3 = nn.ModuleList(
            [EncoderLayer(turb_enc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_enc_layers)])
        self.time_enc_layers_6 = nn.ModuleList(
            [EncoderLayer(time_enc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_enc_layers)])
        self.turb_enc_layers_6 = nn.ModuleList(
            [EncoderLayer(turb_enc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_enc_layers)])


    def forward(self, enc_inputs):
        """
        enc_inputs: [B,134,144,52]
        """
        #enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        '''
        input embedding 1×1 Convolution
        '''
        # enc_time_emb = self.time_embedding(enc_inputs[...,12:].cpu().long().to(enc_inputs.device))
        conv_out = self.conv(enc_inputs.permute(0,3,1,2))
        #conv_out = self.embed_bn(conv_out)
        conv_out = F.relu(conv_out).permute(0,2,3,1)
        # conv_out = conv_out + enc_time_emb
        B,turb,time,dim = conv_out.shape

        '''
        reshape feature map
        '''
        if self.fusion_type==1:
            time_enc_input_1 = conv_out.permute(0,2,1,3).reshape(B,time,-1) #[B,134,144,conv_dim] -> [B,144,134*conv_dim]
            turb_enc_input_1 = conv_out.reshape(B,turb,-1) #[B,134,144,conv_dim] -> [B,134,144*conv_dim

        elif self.fusion_type==2:
            time_enc_input_1 = conv_out.reshape(-1,time,dim) # [B*turb, time,  dim]
            turb_enc_input_1 = conv_out.permute(0,2,1,3).reshape(-1,turb,dim) # [B*time, turb, dim]

        # 计算transformer需要的attn pad mask矩阵, 这里是等长输入和输出，不需要pad mask，所以产生全0矩阵即可
        time_enc_self_attn_mask_1 = torch.zeros((time_enc_input_1.shape[0],time_enc_input_1.shape[1],time_enc_input_1.shape[1]))\
                                    .eq(1).to(enc_inputs.device)  #转化为booltensor
        # [batc_size, src_len, src_len]
        turb_enc_self_attn_mask_1 = torch.zeros((turb_enc_input_1.shape[0],turb_enc_input_1.shape[1],turb_enc_input_1.shape[1]))\
                                    .eq(1).to(enc_inputs.device)
        # [batc_size, src_len, src_len]

        time_enc_outputs_1 = self.time_pos_emb(time_enc_input_1.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        turb_enc_outputs_1 = self.turb_pos_emb(turb_enc_input_1.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]

        time_enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        turb_enc_self_attns = []
        '''
        origin scale
        '''
        time_identity = time_enc_outputs_1
        turb_identity = turb_enc_outputs_1
        for idx in range(self.n_enc_layers):
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            time_enc_outputs_1, time_enc_self_attn = self.time_enc_layers_1[idx](time_enc_outputs_1,
                                                time_enc_self_attn_mask_1)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            turb_enc_outputs_1, turb_enc_self_attn = self.turb_enc_layers_1[idx](turb_enc_outputs_1,
                                                turb_enc_self_attn_mask_1)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            #feature fusion
            time_enc_outputs_1, turb_enc_outputs_1= feature_fusion(
                self.fusion_type, time_enc_outputs_1, turb_enc_outputs_1, (B,turb,time,dim), self.fusion_conv_1)

            time_enc_self_attns.append(time_enc_self_attn)  # 这个只是为了可视化
            turb_enc_self_attns.append(turb_enc_self_attn)

        #残差连接
        time_enc_outputs_1 = time_enc_outputs_1 + time_identity
        turb_enc_outputs_1 = turb_enc_outputs_1 + turb_identity


        '''
        pooling
        '''
        time_enc_input_3 = time_enc_outputs_1.reshape(B, turb, time, dim).permute(0, 3, 1, 2)
        time_enc_outputs_3 = self.maxpool_3(time_enc_input_3).permute(0, 2, 3, 1).reshape(-1, time//3, dim)
        turb_enc_input_3 = turb_enc_outputs_1.reshape(B,time,turb,dim).permute(0, 3, 2, 1)
        turb_enc_outputs_3 = self.maxpool_3(turb_enc_input_3).permute(0, 3, 2, 1).reshape(-1, turb, dim)

        '''
        3 divide scale
        '''
        time_identity = time_enc_outputs_3
        turb_identity = turb_enc_outputs_3
        # 计算transformer需要的attn pad mask矩阵, 这里是等长输入和输出，不需要pad mask，所以产生全0矩阵即可
        time_enc_self_attn_mask_3 = torch.zeros(
            (time_enc_outputs_3.shape[0], time_enc_outputs_3.shape[1], time_enc_outputs_3.shape[1])) \
            .eq(1).to(enc_inputs.device)  # 转化为booltensor
        # [batc_size, src_len, src_len]
        turb_enc_self_attn_mask_3 = torch.zeros(
            (turb_enc_outputs_3.shape[0], turb_enc_outputs_3.shape[1], turb_enc_outputs_3.shape[1])) \
            .eq(1).to(enc_inputs.device)
        # [batc_size, src_len, src_len]

        for idx in range(self.n_enc_layers):
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            time_enc_outputs_3, time_enc_self_attn = self.time_enc_layers_3[idx](time_enc_outputs_3,
                                                time_enc_self_attn_mask_3)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            turb_enc_outputs_3, turb_enc_self_attn = self.turb_enc_layers_3[idx](turb_enc_outputs_3,
                                                turb_enc_self_attn_mask_3)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            #feature fusion
            time_enc_outputs_3, turb_enc_outputs_3= feature_fusion(
                self.fusion_type, time_enc_outputs_3, turb_enc_outputs_3, (B,turb,time//3,dim), self.fusion_conv_1)
        # 残差连接
        time_enc_outputs_3 = time_enc_outputs_3 + time_identity
        turb_enc_outputs_3 = turb_enc_outputs_3 + turb_identity

        '''
        pooling
        '''
        time_enc_input_6 = time_enc_outputs_3.reshape(B, turb, time//3, dim).permute(0, 3, 1, 2)
        time_enc_outputs_6 = self.maxpool_6(time_enc_input_6).permute(0, 2, 3, 1).reshape(-1, time // 6, dim)
        turb_enc_input_6 = turb_enc_outputs_3.reshape(B, time//3, turb, dim).permute(0, 3, 2, 1)
        turb_enc_outputs_6 = self.maxpool_6(turb_enc_input_6).permute(0, 3, 2, 1).reshape(-1, turb, dim)

        '''
        6 divide scale
        '''
        time_identity = time_enc_outputs_6
        turb_identity = turb_enc_outputs_6
        # 计算transformer需要的attn pad mask矩阵, 这里是等长输入和输出，不需要pad mask，所以产生全0矩阵即可
        time_enc_self_attn_mask_6 = torch.zeros(
            (time_enc_outputs_6.shape[0], time_enc_outputs_6.shape[1], time_enc_outputs_6.shape[1])) \
            .eq(1).to(enc_inputs.device)  # 转化为booltensor
        # [batc_size, src_len, src_len]
        turb_enc_self_attn_mask_6 = torch.zeros(
            (turb_enc_outputs_6.shape[0], turb_enc_outputs_6.shape[1], turb_enc_outputs_6.shape[1])) \
            .eq(1).to(enc_inputs.device)
        # [batc_size, src_len, src_len]

        for idx in range(self.n_enc_layers):
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            time_enc_outputs_6, time_enc_self_attn = self.time_enc_layers_6[idx](time_enc_outputs_6,
                                                time_enc_self_attn_mask_6)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            turb_enc_outputs_6, turb_enc_self_attn = self.turb_enc_layers_6[idx](turb_enc_outputs_6,
                                                turb_enc_self_attn_mask_6)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            #feature fusion
            time_enc_outputs_6, turb_enc_outputs_6= feature_fusion(
                self.fusion_type, time_enc_outputs_6, turb_enc_outputs_6, (B, turb, time//6, dim), self.fusion_conv_1 )
        # 残差连接
        time_enc_outputs_6 = time_enc_outputs_6 + time_identity
        turb_enc_outputs_6 = turb_enc_outputs_6 + turb_identity

        return time_enc_outputs_1, turb_enc_outputs_1, \
                time_enc_outputs_3, turb_enc_outputs_3, \
                time_enc_outputs_6, turb_enc_outputs_6, \
                time_enc_self_attns, turb_enc_self_attns


class Decoder(nn.Module):
    def __init__(self,
            config,
            input_dim,
            fusion_type=2,   # 1:token=time/turb * F 2:token = F
            conv_dim = 32,
            d_model = 512,
            d_k = 64,
            d_v = 64,
            n_dec_layers=1,
            n_heads = 8,
            d_ff = 1024
        ):
        super(Decoder, self).__init__()
        self.fusion_type = fusion_type
        self.overlap = config['overlap']
        self.label_len = config['label_len']
        self.n_dec_layers = n_dec_layers
        if fusion_type==1:
            time_dec_dmodel = config['Turbins']*conv_dim
            time_decenc_dmodel= config['Turbins']*conv_dim
            turb_dec_dmodel = config['out_len']*conv_dim
            turb_decenc_dmodel = config['in_len']*conv_dim
            self.time_pos_emb = PositionalEncoding(config['Turbins']*conv_dim)  # Transformer中位置编码时固定的，不需要学习
            self.turb_pos_emb = PositionalEncoding(config['out_len']*conv_dim)
        elif fusion_type==2:
            time_dec_dmodel = conv_dim
            time_decenc_dmodel = conv_dim
            turb_dec_dmodel = conv_dim
            turb_decenc_dmodel = conv_dim
            self.time_pos_emb = PositionalEncoding(conv_dim)  # Transformer中位置编码时固定的，不需要学习
            self.turb_pos_emb = PositionalEncoding(conv_dim)

        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=conv_dim, kernel_size=1)

        self.fusion_conv_1 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=1)
        self.fusion_conv_3 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim , kernel_size=1)
        self.fusion_conv_6 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim , kernel_size=1)

        self.upconv_3 = nn.ConvTranspose2d(in_channels=conv_dim * 2 , out_channels=conv_dim, kernel_size=(1,3),stride=(1,3))
        self.upconv_6 = nn.ConvTranspose2d(in_channels=conv_dim * 2 , out_channels=conv_dim, kernel_size=(1,2),stride=(1,2))

        # 神经网络参数初始化
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        '''
        self.time_dec_layers_1 = nn.ModuleList(
            [DecoderLayer(time_dec_dmodel,time_decenc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.turb_dec_layers_1 = nn.ModuleList(
            [DecoderLayer(turb_dec_dmodel,turb_decenc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.time_dec_layers_3 = nn.ModuleList(
            [DecoderLayer(time_dec_dmodel, time_decenc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.turb_dec_layers_3 = nn.ModuleList(
            [DecoderLayer(turb_dec_dmodel, turb_decenc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.time_dec_layers_6 = nn.ModuleList(
            [DecoderLayer(time_dec_dmodel, time_decenc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_dec_layers)])
        self.turb_dec_layers_6 = nn.ModuleList(
            [DecoderLayer(turb_dec_dmodel, turb_decenc_dmodel, d_k, d_v, n_heads, d_ff) for _ in range(n_dec_layers)])


    def forward(self, dec_inputs,
                time_enc_outputs_1, turb_enc_outputs_1,
                time_enc_outputs_3, turb_enc_outputs_3,
                time_enc_outputs_6, turb_enc_outputs_6):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        '''
        input embedding 1×1 Convolution
        '''

        conv_out = self.conv(dec_inputs.permute(0,3,1,2))
        #conv_out = self.embed_bn(conv_out)
        conv_out = F.relu(conv_out).permute(0,2,3,1)
        # conv_out = conv_out + dec_time_emb
        B, turb, time, dim = conv_out.shape
        device = dec_inputs.device

        '''
        reshape feature map
        '''
        # reshape feature map
        if self.fusion_type == 1:
            time_dec_input = conv_out.permute(0, 2, 1, 3).reshape(B, time, -1)  # [B,134,144,conv_dim] -> [B,144,134*conv_dim]
            turb_dec_input = conv_out.reshape(B, turb, -1)  # [B,134,144,conv_dim] -> [B,134,144*conv_dim

        elif self.fusion_type == 2:
            time_dec_input = conv_out.reshape(-1, time, dim) # [B*turb, time, dim]
            turb_dec_input = conv_out.permute(0, 2, 1, 3).reshape(-1, turb, dim) # [B*time, turb, dim]
        '''
        if self.overlap == 1:
            turb_enc_outputs = turb_enc_outputs.reshape(B,-1,turb,dim)
            turb_enc_outputs = torch.cat((turb_enc_outputs,turb_enc_outputs[:,-self.label_len:,:,:]),1).reshape(-1,turb,dim)
            #print("turb enc outputs:{}, turb dec inputs:{}".format(turb_enc_outputs.shape,turb_dec_input.shape))
        '''
        time_dec_outputs_6 = self.time_pos_emb(time_dec_input.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        turb_dec_outputs_6 = self.turb_pos_emb(turb_dec_input.transpose(0, 1)).transpose(0, 1)


        '''
        6 divide scale
        '''
        time_identity = time_dec_outputs_6
        turb_identity = turb_dec_outputs_6
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        time_dec_selfattn_subsequence_mask_6 = get_attn_subsequence_mask(time_dec_outputs_6). \
            to(device)  # [batch_size, tgt_len, tgt_len]
        turb_dec_selfattn_subsequence_mask_6 = get_attn_subsequence_mask(turb_dec_outputs_6). \
            to(device)  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）,这里pad mask为0,只用subseq maks即可
        time_dec_selfattn_mask_6 = time_dec_selfattn_subsequence_mask_6  # [batch_size, tgt_len, tgt_len]
        turb_dec_selfattn_mask_6 = turb_dec_selfattn_subsequence_mask_6

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        # 计算transformer需要的attn pad mask矩阵,这里不需要pad mask,全0即可
        time_decenc_attn_mask_6 = torch.zeros(
            (time_dec_outputs_6.shape[0], time_dec_outputs_6.shape[1], time_enc_outputs_6.shape[1])) \
            .eq(1).to(device)  # 转化为booltensor
        # [batc_size, tgt_len, src_len]
        turb_decenc_attn_mask_6 = torch.zeros(
            (turb_dec_outputs_6.shape[0], turb_dec_outputs_6.shape[1], turb_enc_outputs_6.shape[1])) \
            .eq(1).to(device)

        # dec_self_attns, dec_enc_attns = [], []
        for idx in range(self.n_dec_layers):
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            time_dec_outputs_6, time_dec_selfattn, time_decenc_attn = self.time_dec_layers_6[idx](
                time_dec_outputs_6, time_enc_outputs_6, time_dec_selfattn_mask_6,
                time_decenc_attn_mask_6)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            turb_dec_outputs_6, turb_dec_self_attn, turb_decenc_attn = self.turb_dec_layers_6[idx](
                turb_dec_outputs_6, turb_enc_outputs_6, turb_dec_selfattn_mask_6,
                turb_decenc_attn_mask_6)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            # feature fusion
            time_dec_outputs_6, turb_dec_outputs_6 = feature_fusion(
                self.fusion_type, time_dec_outputs_6, turb_dec_outputs_6, (B, turb, time, dim), self.fusion_conv_1)
        #残差连接
        time_dec_outputs_6 = time_dec_outputs_6 + time_identity
        turb_dec_outputs_6 = turb_dec_outputs_6 + turb_identity

        '''
        up-conv
        '''
        time_dec_outputs_3, turb_dec_outputs_3 = up_conv(time_dec_outputs_6, time_enc_outputs_6, (B, turb, time, dim), self.upconv_6)

        '''
        3 divide scale
        '''
        time_identity = time_dec_outputs_3
        turb_identity = turb_dec_outputs_3
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        time_dec_selfattn_subsequence_mask_3 = get_attn_subsequence_mask(time_dec_outputs_3). \
            to(device)  # [batch_size, tgt_len, tgt_len]
        turb_dec_selfattn_subsequence_mask_3 = get_attn_subsequence_mask(turb_dec_outputs_3). \
            to(device)  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）,这里pad mask为0,只用subseq maks即可
        time_dec_selfattn_mask_3 = time_dec_selfattn_subsequence_mask_3  # [batch_size, tgt_len, tgt_len]
        turb_dec_selfattn_mask_3 = turb_dec_selfattn_subsequence_mask_3

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        # 计算transformer需要的attn pad mask矩阵,这里不需要pad mask,全0即可
        time_decenc_attn_mask_3 = torch.zeros(
            (time_dec_outputs_3.shape[0], time_dec_outputs_3.shape[1], time_enc_outputs_3.shape[1])) \
            .eq(1).to(device)  # 转化为booltensor
        # [batc_size, tgt_len, src_len]
        turb_decenc_attn_mask_3 = torch.zeros(
            (turb_dec_outputs_3.shape[0], turb_dec_outputs_3.shape[1], turb_enc_outputs_3.shape[1])) \
            .eq(1).to(device)

        # dec_self_attns, dec_enc_attns = [], []
        for idx in range(self.n_dec_layers):
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            time_dec_outputs_3, time_dec_selfattn, time_decenc_attn = self.time_dec_layers_3[idx](
                time_dec_outputs_3, time_enc_outputs_3, time_dec_selfattn_mask_3,
                time_decenc_attn_mask_3)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            turb_dec_outputs_3, turb_dec_self_attn, turb_decenc_attn = self.turb_dec_layers_3[idx](
                turb_dec_outputs_3, turb_enc_outputs_3, turb_dec_selfattn_mask_3,
                turb_decenc_attn_mask_3)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            # feature fusion
            time_dec_outputs_3, turb_dec_outputs_3 = feature_fusion(
                self.fusion_type, time_dec_outputs_3, turb_dec_outputs_3, (B, turb, time * 2, dim), self.fusion_conv_1 )
        # 残差连接
        time_dec_outputs_3 = time_dec_outputs_3 + time_identity
        turb_dec_outputs_3 = turb_dec_outputs_3 + turb_identity

        '''
        up-conv
        '''
        time_dec_outputs_1, turb_dec_outputs_1 = up_conv(time_dec_outputs_3, time_enc_outputs_3, (B, turb, time * 2, dim), self.upconv_3)


        '''
        origin scale
        '''
        time_identity = time_dec_outputs_1
        turb_identity = turb_dec_outputs_1
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        time_dec_selfattn_subsequence_mask_1 = get_attn_subsequence_mask(time_dec_outputs_1). \
            to(device)  # [batch_size, tgt_len, tgt_len]
        turb_dec_selfattn_subsequence_mask_1 = get_attn_subsequence_mask(turb_dec_outputs_1). \
            to(device)  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）,这里pad mask为0,只用subseq maks即可
        time_dec_selfattn_mask_1 = time_dec_selfattn_subsequence_mask_1  # [batch_size, tgt_len, tgt_len]
        turb_dec_selfattn_mask_1 = turb_dec_selfattn_subsequence_mask_1

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        # 计算transformer需要的attn pad mask矩阵,这里不需要pad mask,全0即可
        time_decenc_attn_mask_1 = torch.zeros(
            (time_dec_outputs_1.shape[0], time_dec_outputs_1.shape[1], time_enc_outputs_1.shape[1])) \
            .eq(1).to(device)  # 转化为booltensor
        # [batc_size, tgt_len, src_len]
        turb_decenc_attn_mask_1 = torch.zeros(
            (turb_dec_outputs_1.shape[0], turb_dec_outputs_1.shape[1], turb_enc_outputs_1.shape[1])) \
            .eq(1).to(device)

        # dec_self_attns, dec_enc_attns = [], []
        for idx in range(self.n_dec_layers):
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            time_dec_outputs_1, time_dec_selfattn, time_decenc_attn = self.time_dec_layers_1[idx](
                time_dec_outputs_1, time_enc_outputs_1, time_dec_selfattn_mask_1,
                time_decenc_attn_mask_1)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            turb_dec_outputs_1, turb_dec_self_attn, turb_decenc_attn = self.turb_dec_layers_1[idx](
                turb_dec_outputs_1, turb_enc_outputs_1, turb_dec_selfattn_mask_1,
                turb_decenc_attn_mask_1)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            # feature fusion
            time_dec_outputs_1, turb_dec_outputs_1 = feature_fusion(
                self.fusion_type, time_dec_outputs_1, turb_dec_outputs_1, (B, turb, time*6, dim), self.fusion_conv_1)
        # 残差连接
        time_dec_outputs_1 = time_dec_outputs_1 + time_identity
        turb_dec_outputs_1 = turb_dec_outputs_1 + turb_identity

        '''
        concate
        '''
        time_dec_outputs_1 = torch.cat((time_enc_outputs_1, time_dec_outputs_1), -1)
        turb_dec_outputs_1 = torch.cat((turb_enc_outputs_1, turb_dec_outputs_1), -1)
        time_dec_outputs_3 = torch.cat((time_enc_outputs_3, time_dec_outputs_3), -1)
        turb_dec_outputs_3 = torch.cat((turb_enc_outputs_3, turb_dec_outputs_3), -1)
        time_dec_outputs_6 = torch.cat((time_enc_outputs_6, time_dec_outputs_6), -1)
        turb_dec_outputs_6 = torch.cat((turb_enc_outputs_6, turb_dec_outputs_6), -1)

        # dec_outputs: [batch_size, turb, time, conv_dim*2]
        if self.fusion_type==1:
            outputs_1 = torch.cat((time_dec_outputs_1.reshape(B,time,turb,-1).permute(0,2,1,3), turb_dec_outputs_1.reshape(B,turb,time,-1)),-1)
        elif self.fusion_type==2:
            #outputs_1 = torch.cat((time_dec_outputs_1.reshape(B,turb,time*6,-1),turb_dec_outputs_1.reshape(B,time*6,turb,-1).permute(0,2,1,3)),-1)
            #outputs_3 = torch.cat((time_dec_outputs_3.reshape(B, turb, time*2, -1),turb_dec_outputs_3.reshape(B, time*2, turb, -1).permute(0, 2, 1, 3)), -1)
            #outputs_6 = torch.cat((time_dec_outputs_6.reshape(B, turb, time, -1),turb_dec_outputs_6.reshape(B, time, turb, -1).permute(0, 2, 1, 3)), -1)
            outputs_1 = time_dec_outputs_1.reshape(B, turb, time * 6, -1)
            outputs_3 = time_dec_outputs_3.reshape(B, turb, time * 2, -1)
            outputs_6 = time_dec_outputs_6.reshape(B, turb, time, -1)

        return outputs_1,outputs_3,outputs_6

class Transformer(nn.Module):
    def __init__(self,
        # Transformer Parameters
        config,
        enc_input_dim,
        dec_input_dim,
        #fusion_type=2,
        #conv_dim = 32,
        #d_model = 512,  # Embedding Size（token embedding和position编码的维度）
        #d_ff = 2048,  # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
        #d_k = 64,
        #d_v = 64,  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
        #n_enc_layers = 6,  # number of Encoder  Layer（Block的个数）
        #n_dec_layers = 6,  #number of Decoder Layer
        #n_heads = 8,  # number of heads in Multi-Head Attention（有几套头）
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config,input_dim=enc_input_dim,fusion_type=config['fusion_type'],conv_dim=config['conv_dim'],
                               d_model=config['d_model'],d_k=config['d_k'],d_v=config['d_k'],n_enc_layers=config['n_enc_layers'],
                               n_heads=config['n_heads'],d_ff=config['d_ff'])
        self.decoder = Decoder(config,input_dim=dec_input_dim,fusion_type=config['fusion_type'],conv_dim=config['conv_dim'],
                               d_model=config['d_model'],d_k=config['d_k'],d_v=config['d_k'],n_dec_layers=config['n_dec_layers'],
                               n_heads=config['n_heads'],d_ff=config['d_ff'])
        # zy
        self.dropout = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(config['conv_dim']*2, config['conv_dim'])
        self.projection = nn.Linear(config['conv_dim'] * 2 , 1)
        self.regress_conv = nn.Conv2d(in_channels=config['conv_dim']*2, out_channels=1, kernel_size=1)

    def forward(self, enc_inputs, dec_inputs):
        """Transformers的输入：两个序列
        enc_inputs: [B,turb,src_time,src_dim]
        dec_inputs: [B,turb,tgt_time,tgt_dim]
        """

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        time_enc_outputs_1, turb_enc_outputs_1, \
        time_enc_outputs_3, turb_enc_outputs_3, \
        time_enc_outputs_6, turb_enc_outputs_6, \
        time_enc_self_attns, turb_enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs_1, dec_outputs_3, dec_outputs_6  = self.decoder(dec_inputs,
                                    time_enc_outputs_1, turb_enc_outputs_1,
                                    time_enc_outputs_3, turb_enc_outputs_3,
                                    time_enc_outputs_6, turb_enc_outputs_6)

        dec_logits_1 = self.projection(self.dropout(dec_outputs_1)).squeeze(-1)
        dec_logits_3 = self.projection(self.dropout(dec_outputs_3)).squeeze(-1)
        dec_logits_6 = self.projection(self.dropout(dec_outputs_6)).squeeze(-1)

        return dec_logits_1, dec_logits_3, dec_logits_6 #[B,turb,time]


class HSTTN(nn.Module):
    def __init__(
            self,
            config
                 ):
        super(HSTTN,self).__init__()

        #self.turb_embed = nn.Embedding(turbins, 10)
        self.overlap = config['overlap']
        self.transformer = Transformer(config, enc_input_dim=config['enc_in'], dec_input_dim=config['dec_in'])
        self.pred_len = config['pred_len']


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # [B, 134, 144, 10]
        B, turb, input_len, in_dim = x_enc.shape
        _, _, output_len, out_dim = x_dec.shape
        #[B, 134, 144, 10]
        #enc_turb_emb = self.turb_embed(batch_sparse_x[:,:,:,2])
        #dec_turb_emb = self.turb_embed(batch_sparse_y[:,:,:,2])

        #x_mark_dec = torch.zeros(x_mark_dec.shape).to(x_enc.device)
        # 构建encoder和decoder的输入
        enc_input = torch.cat((x_enc, x_mark_enc), -1)  # [B,turb,144,attr+time]
        if self.overlap == 0:
            dec_input = torch.cat((x_dec[:, :, -self.pred_len:, :], x_mark_dec[:, :, -self.pred_len:, :]), -1)  # [B,turb,144,attr+time]
            #dec_input = x_mark_dec[:, :, -self.pred_len:, :]
        else:
            dec_input = torch.cat((x_dec,x_mark_dec),-1)
        # dec_input = batch_sparse_y
        # print("enc_input shape:{}, dec_input shape:{}".format(enc_input.shape,dec_input.shape))

        # transformer
        outputs_1, outputs_3, outputs_6 = self.transformer(enc_input, dec_input)

        return outputs_1, outputs_3, outputs_6  # [B,turb,144]