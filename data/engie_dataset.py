#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/7 10:24
# @Author : ZhangYang
# @Site : 
# @File : engie_dataset.py
# @Software: PyCharm
from torch.utils.data import Dataset
import os
import time
import datetime
import numpy as np
import pandas as pd

def time2obj(time_raw):
    sj = time_raw.split('+')[0] #2013-01-01 00:00:00+01:00
    day_sj, time_sj = sj.split(' ')
    day_obj = time.strptime(day_sj, "%Y-%m-%d")
    time_obj = time.strptime(time_sj, "%H:%M:%S")
    return day_obj, time_obj

def func_add_year(x):
    day_obj, time_obj = time2obj(x)
    year = day_obj.tm_year
    return year
def func_add_mon(x):
    day_obj, time_obj = time2obj(x)
    mon = day_obj.tm_mon
    return mon
def func_add_wday(x):
    day_obj, time_obj = time2obj(x)
    weekday = day_obj.tm_wday
    return weekday
def func_add_day(x):
    day_obj, time_obj = time2obj(x)
    day = day_obj.tm_mday
    return day
def func_add_hour(x):
    day_obj, time_obj = time2obj(x)
    hour = time_obj.tm_hour
    return hour
def func_add_min(x):
    day_obj, time_obj = time2obj(x)
    minute = time_obj.tm_min
    return minute

class ENGIEDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    1296 days for training,
                        180 days for validation,
                        360 days for testing
    """
    def __init__(
        self,
        data_path,
        filename='new_engie.csv',
        locationfile='location.csv',
        flag='train',
        multi_scale=False,
        in_len=144,
        out_len=144,
        label_len=144,
        timeenc=1,
        day_len=24 * 6,
        Turbins=4,
        train_days=1296,  # 1296 days
        val_days=180,  # 180 days
        test_days=360,  # 360 days
        total_days=1836,  # 1836 days
    ):
        super().__init__()

        # initialization
        self.unit_size = day_len
        self.input_len = in_len
        self.output_len = out_len
        self.label_len = label_len
        self.start_col = 0
        self.Turbins = Turbins

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename
        self.locationfile = locationfile
        self.multi_scale = multi_scale

        self.total_size = total_days * self.unit_size
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size
        self.__read_location__()
        self.__read_data__()


    def __read_data__(self):
        #read wind power data
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        attr_data, time_data = self.build_data(df_data)
        print(f"attr_data_shape: {attr_data.shape}")
        print(f"sparse_data_shape: {time_data.shape}")
        self.attr_data = attr_data
        self.time_data = time_data

    def __read_location__(self):
        #read location data

        df_location = pd.read_csv(os.path.join(self.data_path, self.locationfile))
        location = df_location.values[:,1:]
        #print(location)
        mean = np.mean(location, axis=0, keepdims=True,dtype=float)
        std = np.std(location, axis=0, keepdims=True, dtype=float)
        #print("location mean:{}, std:{}".format(mean.shape,std.shape))#[1,2]
        location = (location-mean)/std
        self.location = location #(4,2)

    def data_preprocess(self, df_data):
        """
        1. 增加time feature
        2. 将nan 置 0
        3. 将prtv和patv小于0置0
        :param df_data:
        :return:
        """
        #print(df_data['Wind_turbine_name'].value_counts())
        R80711 = df_data[df_data['Wind_turbine_name'].str.contains('R80711')][:264384]
        R80721 = df_data[df_data['Wind_turbine_name'].str.contains('R80721')][:264384]
        R80736 = df_data[df_data['Wind_turbine_name'].str.contains('R80736')][:264384]
        R80790 = df_data[df_data['Wind_turbine_name'].str.contains('R80790')]
        df_data = pd.concat([R80736,R80721,R80711,R80790],ignore_index=True) #根据location文件的顺序拼接
        #print(df_data.shape) #1057536,28

        feature_name = [
            n for n in df_data.columns
            if 'min' not in n and 'max' not in n and "std" not in n and 'name' not in n and
               'time' not in n and 'P_avg' != n and 'Q_avg' != n and 'Na_c_avg' not in n and
               'Pas_avg' not in n and 'Wa_c_avg' not in n and 'Va_avg' not in n and 'Va1_avg' not in n and 'Va2_avg' not in n
        ]
        feature_name.append('Q_avg')
        feature_name.append('P_avg')
        new_df_data = df_data[feature_name]  #28 columns

        # add time attr
        year = df_data['Date_time'].apply(func_add_year)
        mon = df_data['Date_time'].apply(func_add_mon)
        weekday = df_data['Date_time'].apply(func_add_wday)
        day = df_data['Date_time'].apply(func_add_day)
        hour = df_data['Date_time'].apply(func_add_hour)
        minute = df_data['Date_time'].apply(func_add_min)
        # 插入时间并归一化
        new_df_data.insert(0, 'year', year) # 2013年开始
        new_df_data.insert(1, 'mon', mon )
        new_df_data.insert(2, 'weedday', weekday )
        new_df_data.insert(3, 'day', day )
        new_df_data.insert(4, 'hour', hour )
        new_df_data.insert(5, 'min', minute //10 )
       # print(new_df_data.columns, len(new_df_data.columns)) #['year', 'mon', 'weedday', 'day', 'hour', 'min', 'Ba_avg', 'Rt_avg',
       # 'DCs_avg', 'Cm_avg', 'S_avg', 'Cosphi_avg', 'Ds_avg', 'Db1t_avg',
       # 'Db2t_avg', 'Dst_avg', 'Gb1t_avg', 'Gb2t_avg', 'Git_avg', 'Gost_avg',
       # 'Ya_avg', 'Yt_avg', 'Ws1_avg', 'Ws2_avg', 'Ws_avg', 'Wa_avg', 'Ot_avg',
       # 'Nf_avg', 'Nu_avg', 'Rs_avg', 'Rbt_avg', 'Rm_avg', 'Q_avg', 'P_avg'] 34

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        #将nan置0
        # nan_sum = np.sum(np.isnan(new_df_data.values))
        # print(nan_sum) #196546
        # new_df_data = new_df_data.replace(
        #     to_replace=np.nan, value=0, inplace=False)
        new_df_data.fillna(method='ffill', axis=1, inplace=True, limit=100)

        # print(nan_sum) #196546
        #将Prtv,Patv小于0置0
        new_df_data.loc[new_df_data['Prtv']<0,('Prtv')] = 0
        new_df_data.loc[new_df_data['Patv'] < 0, ('Patv')] = 0
        new_df_data.loc[(raw_df_data['Patv'] < 0) | \
                       ((raw_df_data['Patv'] == 0) & (raw_df_data['Wspd'] > 2.5)) | \
                       ((raw_df_data['Pab1'] > 89) | (raw_df_data['Pab2'] > 89) | (raw_df_data['Pab3'] > 89)) | \
                       ((raw_df_data['Wdir'] < -180) | (raw_df_data['Wdir'] > 180) | (raw_df_data['Ndir'] < -720) |
                        (raw_df_data['Ndir'] > 720)),
                        ('Patv')]=-1
        #rtv_neg_sum = np.sum(new_df_data['Prtv']<0)
        #print("rtv_neg_sum:{}".format(rtv_neg_sum))
        #atv_neg_sum = np.sum(new_df_data['Patv']==-1)
        #print('atv_neg_sum:{}'.format(atv_neg_sum))
        #cond = (~invalid_cond) & (~nan_cond)

        return new_df_data, raw_df_data

    def build_data(self, df_data):
        cols_data = df_data.columns
        raw_df_data = self.raw_df_data

        raw_data = raw_df_data.values
        raw_data = np.reshape(
            raw_data, [self.Turbins, self.total_size, len(cols_data)])

        data = df_data.values #(1057536, 34)
        data = np.reshape(data,
                   [self.Turbins, self.total_size, len(cols_data)]) #[4, 264384, 34]

        #划分train,valid,test数据边界
        border1s = [
            0,
            self.train_size - self.input_len,
            self.train_size + self.val_size - self.input_len
        ]
        border2s = [
            self.train_size,
            self.train_size + self.val_size,
            self.train_size + self.val_size + self.test_size
        ]

        #只求了train的数据均值和标准差
        self.data_mean = np.mean(
                data[:, border1s[0]:border2s[0], :],
                axis=1,
                keepdims=True)
        self.data_scale = np.std(
                data[:, border1s[0]:border2s[0], :],
                axis=1,
                keepdims=True)
        self.data_mean = np.around(self.data_mean,4)
        self.data_scale = np.around(self.data_scale,4)
        # print("mean shape:{}".format(self.data_mean.shape)) #(4,1,34)
        # print("std shape:{}".format(self.data_scale.shape)) #(4,1,34)

        #在此处做归一化
        data[:, :, :] = (data[:, :, :] - self.data_mean) / self.data_scale  # [4, 264384, 34]
        time_data = data[:,:,:6] #[[4, 264384, 6]]
        attr_data = data[:,:,6:] #[[4, 264384, 28]]
        location = np.expand_dims(self.location, 1).repeat(data.shape[1], axis=1)  # 复制矩阵 [4, 264384, 2]
        attr_data = np.concatenate((location,attr_data),axis=-1) #[4, 264384, 30]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        #print("border1:{}, border2:{}".format(border1,border2))

        self.raw_df = []
        for turb_id in range(self.Turbins):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id, border1 + self.input_len:border2],
                    columns=cols_data))
        print("raw df shape:{}".format(self.raw_df[0].shape))
        # 返回train,valid,test对应的数据
        attr_data = attr_data[:, border1:border2, :]
        time_data = time_data[:, border1:border2, :]
        #location = location[:, border1:border2, :]

        return attr_data,time_data
    def get_raw_df(self):
        return self.raw_df

    def __len__(self):
        return self.attr_data.shape[1] - self.input_len - self.output_len + 1

    def select_item(self,index, interval):
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.output_len
        # print("s_begin: {}, s_end: {}, r_begin: {}, r_end: {}".format(s_begin,s_end,r_begin,r_end))
        # origin scale data
        attr = self.attr_data[:, s_begin:s_end, :]  # [134,144,10]
        sparse_x = self.time_data[:, s_begin:s_end, :]  # [134,144,2]
        sparse_y = self.time_data[:, r_begin:r_end, :]  # [134,288,2]

        y = self.attr_data[:, r_begin:r_end, :]  # [134,288,10]

        # multi-scale data
        seq_x = np.zeros([attr.shape[0], attr.shape[1]//interval, attr.shape[2]])
        seq_sparse_x = np.zeros([sparse_x.shape[0], sparse_x.shape[1]//interval, sparse_x.shape[2]])
        seq_sparse_y = np.zeros([sparse_y.shape[0], sparse_y.shape[1]//interval, sparse_y.shape[2]])
        seq_y = np.zeros([y.shape[0], y.shape[1]//interval, y.shape[2]])
        #print("seq x shape:{}, sparse x shape:{}, sparse y shape:{}, seq y shape:{}".format(seq_x.shape,seq_sparse_x.shape,seq_sparse_y.shape,seq_y.shape))
        #print("interval:{}, seq x shape[1]:{}".format(interval,seq_x.shape[1]))
        for i in range(seq_x.shape[1]):
            seq_x[:,i,:] = np.mean(attr[:,i*interval:i*interval+interval,:], axis=1)
            seq_sparse_x[:,i,:] = sparse_x[:,i*interval+interval-1,:]
        for i in range(seq_y.shape[1]):
            seq_sparse_y[:, i, :] = sparse_y[:, i * interval + interval - 1, :]
            seq_y[:,i,:] = np.mean(y[:,i*interval:i*interval+interval,:], axis=1)

        #print("seq_attr, seq_sparse_x, seq_sparse_y, seq_location_x, seq_location_y, seq_y :",
        #      seq_attr.shape, seq_sparse_x.shape, seq_sparse_y.shape, seq_location_x.shape, seq_location_y.shape, seq_y.shape)
        return seq_x, seq_y, seq_sparse_x, seq_sparse_y

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        seq_x_10, seq_y_10, seq_sparse_x_10, seq_sparse_y_10 = self.select_item( index, interval=1)
        seq_x_30, seq_y_30, seq_sparse_x_30, seq_sparse_y_30 = self.select_item( index, interval=3)
        seq_x_60, seq_y_60, seq_sparse_x_60, seq_sparse_y_60 = self.select_item( index, interval=6)
        # 打乱turbin顺序
        '''
        if self.flag == "train":
            perm = np.arange(0, seq_attr.shape[0])
            np.random.shuffle(perm)
            return seq_attr[perm],seq_sparse_x[perm],seq_sparse_y[perm], seq_location_x[perm], seq_location_y[perm], seq_y[perm]
        else:
            return seq_attr, seq_sparse_x, seq_sparse_y, seq_location_x, seq_location_y, seq_y
        '''
        if self.multi_scale:
            return seq_x_10, seq_sparse_x_10, seq_sparse_y_10,  seq_y_10, \
                   seq_x_30, seq_sparse_x_30, seq_sparse_y_30,  seq_y_30, \
                   seq_x_60, seq_sparse_x_60, seq_sparse_y_60,  seq_y_60
        else:
            return seq_x_10, seq_y_10, seq_sparse_x_10, seq_sparse_y_10,


if __name__ == "__main__":
    tmp_dataset = ENGIEDataset(
        data_path = '/data/zhangyang/WPF/engie/',
        flag='train',
        locationfile='location.csv',
        multi_scale=False,
        )