import torch
import torch.nn.functional as F
from torch import tensor

from torch.utils.data import Dataset

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import os
import time
import datetime
import numpy as np
import pandas as pd

def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj


def time2int(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    time_int = int(time.mktime(data_sj))
    return time_int


def int2time(t):
    timestamp = datetime.datetime.fromtimestamp(t)
    return timestamp.strftime('"%H:%M"')


def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e


def func_add_h(x):
    time_obj = time2obj(x)
    hour_e = time_obj.tm_hour
    return hour_e

class KDDCUPDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    155 days for training,
                        30 days for validation,
                        60 days for testing
    """
    def __init__(
        self,
        data_path,
        filename='wtbdata_245days.csv',
        locationfile='sdwpf_turb_location.CSV',
        flag='train',
        multi_scale=False,
        in_len=144,
        out_len=144,
        label_len=144,
        timeenc=1,
        day_len=24 * 6,
        Turbins=134,
        train_days=155,  # 155 days
        val_days=30,  # 30 days
        test_days=60,  # 60 days
        total_days=245,  # 245 days
        theta=0.9,
    ):
        super().__init__()

        # initialization
        self.unit_size = day_len
        self.input_len = in_len
        self.output_len = out_len
        self.label_len = label_len
        self.start_col = 0
        self.Turbins = Turbins
        self.theta = theta

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
        mean = np.mean(location, axis=0, keepdims=True)
        std = np.std(location, axis=0, keepdims=True)
        #print("location mean:{}, std:{}".format(mean.shape,std.shape))
        location = (location-mean)/std
        self.location = location #(134,2)

    def data_preprocess(self, df_data):
        """
        1. 增加time feature
        2. 将nan 置 0
        3. 将prtv和patv小于0置0
        :param df_data:
        :return:
        """
        feature_name = [
            n for n in df_data.columns
            if 'Day' not in n and 'Tmstamp' not in n and "TurbID" not in n
        ]
        #print("feature name:{}".format(feature_name)) #10 columns

        new_df_data = df_data[feature_name]  #10 columns
        # add time attr
        t = df_data['Tmstamp'].apply(func_add_t) #0-143
        new_df_data.insert(0, 'time', t)

        month = (df_data['Day'].apply(lambda x: x // 31)) / 11.0 - 0.5 #0-11 数据集小于1年
        weekday = (df_data['Day'].apply(lambda x: x % 7)) / 6.0 - 0.5 #0-6
        day = (df_data['Day'].apply(lambda x: x % 31)) / 30.0 - 0.5 #0-30
        hour = (new_df_data['time'].apply(lambda x: x//6)) / 23.0 - 0.5 #0-23
        minute = new_df_data['time'].apply(lambda x: x % 6) / 5.0 - 0.5#0-5
        new_df_data.insert(0, 'minute', minute)
        new_df_data.insert(0, 'hour', hour)
        new_df_data.insert(0, 'weekday', weekday)
        new_df_data.insert(0, 'day', day)
        new_df_data.insert(0,'month',month)

        new_df_data.drop(columns='time',inplace=True)
        #print("new columns:{}".format(new_df_data.columns)) #['month', 'day', 'weekday', 'hour', 'minute', 'Wspd', 'Wdir', 'Etmp',
        #'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
        #new_df_data.to_csv('./output/new_df_data.csv')

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data.copy(deep=True)
        #将nan置0
        # new_df_data = new_df_data.replace(
        #     to_replace=np.nan, value=0, inplace=False)
        new_df_data.fillna(method='ffill', axis=1, inplace=True, limit=100)
        #将Prtv,Patv小于0置0
        new_df_data.loc[raw_df_data['Prtv']<0,('Prtv')] = -100
        new_df_data.loc[raw_df_data['Patv'] < 0, ('Patv')] = -100
        new_df_data.loc[(raw_df_data['Patv'] < 0) | \
                       ((raw_df_data['Patv'] == 0) & (raw_df_data['Wspd'] > 2.5)) | \
                       ((raw_df_data['Pab1'] > 89) | (raw_df_data['Pab2'] > 89) | (raw_df_data['Pab3'] > 89)) | \
                       ((raw_df_data['Wdir'] < -180) | (raw_df_data['Wdir'] > 180) | (raw_df_data['Ndir'] < -720) |
                        (raw_df_data['Ndir'] > 720)),
                        ('Patv')]=-100
        #rtv_neg_sum = np.sum(new_df_data['Prtv']<0)
        #print("rtv_neg_sum:{}".format(rtv_neg_sum))
        #atv_neg_sum = np.sum(new_df_data['Patv']==-1)
        #print('atv_neg_sum:{}'.format(atv_neg_sum))
        #cond = (~invalid_cond) & (~nan_cond)

        return new_df_data, raw_df_data

    def build_data(self, df_data):
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]

        raw_data = raw_df_data.values
        raw_data = np.reshape(
            raw_data, [self.Turbins, self.total_size, len(cols_data)])

        data = df_data.values #[4727520,12]
        data = np.reshape(data,
                   [self.Turbins, self.total_size, len(cols_data)]) #[134,35280,12]


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
        #print("border1s: ", border1s) #[0,30672,32976]
        #print("border2s: ",border2s) #[30816,33120,35280]

        #只求了train的数据均值和标准差
        self.data_mean = np.mean(
                data[:, border1s[0]:border2s[0], -10:],
                axis=1,
                keepdims=True)
        self.data_scale = np.std(
                data[:, border1s[0]:border2s[0], -10:],
                axis=1,
                keepdims=True)
        self.data_mean = np.around(self.data_mean,4)
        self.data_scale = np.around(self.data_scale,4)
        #print("mean:{}, std:{}".format(self.data_mean[10,0,-1],self.data_scale[10,0,-1]))

        #print("mean shape:{}".format(self.data_mean.shape)) #(134,1,10)
        #print("std shape:{}".format(self.data_scale.shape)) #(134,1,10)

        #在此处做归一化
        data[:, :, -10:] = (data[:, :, -10:] - self.data_mean) / self.data_scale  # [134,35280,12]
        time_data = data[:,:,:5] #[134,35280,5] month,day,weedkay,hour,minute
        attr_data = data[:,:,5:] #[134,35280,10]
        location = np.expand_dims(self.location, 1).repeat(data.shape[1], axis=1)  # 复制矩阵 [134,35280,2]
        attr_data = np.concatenate((location,attr_data),axis=-1) #[134,35280,12]

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
            perm = np.arange(0, seq_x_10.shape[0])
            np.random.shuffle(perm)
            if self.multi_scale:
                return seq_x_10[perm], seq_sparse_x_10[perm], seq_sparse_y_10[perm], seq_y_10[perm], \
                       seq_x_30[perm], seq_sparse_x_30[perm], seq_sparse_y_30[perm], seq_y_30[perm], \
                       seq_x_60[perm], seq_sparse_x_60[perm], seq_sparse_y_60[perm], seq_y_60[perm]
            else:
                return seq_x_10[perm], seq_y_10[perm], seq_sparse_x_10[perm], seq_sparse_y_10[perm],
        else:
            if self.multi_scale:
                return seq_x_10, seq_sparse_x_10, seq_sparse_y_10, seq_y_10, \
                       seq_x_30, seq_sparse_x_30, seq_sparse_y_30, seq_y_30, \
                       seq_x_60, seq_sparse_x_60, seq_sparse_y_60, seq_y_60
            else:
                return seq_x_10, seq_y_10, seq_sparse_x_10, seq_sparse_y_10,

        '''
        if self.multi_scale:
            return seq_x_10, seq_sparse_x_10, seq_sparse_y_10,  seq_y_10, \
                   seq_x_30, seq_sparse_x_30, seq_sparse_y_30,  seq_y_30, \
                   seq_x_60, seq_sparse_x_60, seq_sparse_y_60,  seq_y_60
        else:
            return seq_x_10, seq_y_10, seq_sparse_x_10, seq_sparse_y_10,



if __name__ == "__main__":
    pass