from data.kddcup_dataset import KDDCUPDataset
from exp.exp_basic import Exp_Basic
from models.HSTTN.model import HSTTN

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import regressor_detailed_scores, all_turb_multi_interval_scores, show_metric_scores
from utils.loss import MAE, masked_mae_loss,masked_mse_loss

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_kddcup_HSTTN(Exp_Basic):
    def __init__(self, args):
        super(Exp_kddcup_HSTTN, self).__init__(args)
        self._get_data()
    
    def _build_model(self):

        model_config = self.args.__dict__.copy()
        model_config['conv_dim'] = self.args.d_model
        model_config['in_len'] = self.args.seq_len
        model_config['out_len'] = self.args.pred_len
        model_config['n_enc_layers'] = self.args.e_layers
        model_config['n_dec_layers'] = self.args.d_layers
        model_config['fusion_type'] = 2
        if self.args.model == "HSTTN":
            model = HSTTN(model_config).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        args = self.args

        timeenc = 0 if args.embed!='timeF' else 1
        train_dataset = KDDCUPDataset(
            args.data_path,
            flag='train',
            multi_scale=args.multiscale,
            in_len=args.seq_len,
            out_len=args.pred_len,
            label_len=args.label_len,
            timeenc=timeenc,
            total_days=args.total_days,
            train_days=args.train_days,
            val_days=args.val_days,
            test_days=args.test_days)
        val_dataset = KDDCUPDataset(
            args.data_path,
            flag='val',
            multi_scale=args.multiscale,
            in_len=args.seq_len,
            out_len=args.pred_len,
            label_len=args.label_len,
            timeenc=timeenc,
            total_days=args.total_days,
            train_days=args.train_days,
            val_days=args.val_days,
            test_days=args.test_days)
        test_dataset = KDDCUPDataset(
            args.data_path,
            flag='test',
            multi_scale=args.multiscale,
            in_len=args.seq_len,
            out_len=args.pred_len,
            label_len=args.label_len,
            timeenc=timeenc,
            total_days=args.total_days,
            train_days=args.train_days,
            val_days=args.val_days,
            test_days=args.test_days)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.test_loader = test_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        #model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        #model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim
    
    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion =  nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = MAE()
        elif self.args.loss == 'mask_mae':
            criterion = masked_mae_loss
        elif self.args.loss == 'mask_mse' :
            criterion = masked_mse_loss
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                pred_10, pred_30, pred_60, true_10, true_30, true_60 = self._process_one_batch(batch)
                # print("pred shape:{}, true shape:{}".format(pred.shape,true.shape)) #[B,134,144]
                loss_10 = criterion(pred_10, true_10)
                # loss_30 = criterion(pred_30, true_30)
                # loss_60 = criterion(pred_60, true_60)
                # loss = loss_10 + loss_30 + loss_60
                loss = loss_10

                preds.append(pred_10.detach().cpu().numpy())
                trues.append(true_10.detach().cpu().numpy())

                total_loss.append(loss.detach().cpu())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        # print('valid shape:', preds.shape, trues.shape, self.val_dataset.get_raw_df()[0].shape)  #
        # 逆标准化
        mean = np.expand_dims(self.train_dataset.data_mean[:, :, -1], 0)
        std = np.expand_dims(self.train_dataset.data_scale[:, :, -1], 0)
        #print("mean shape:{}, std shape:{}".format(mean.shape, std.shape))  # [1,134,1]
        preds = preds * std + mean
        trues = trues * std + mean

        preds = np.expand_dims(preds, -1).transpose([1, 0, 2, 3])
        trues = np.expand_dims(trues, -1).transpose([1, 0, 2, 3])
        sep_scores = regressor_detailed_scores(preds, trues, self.val_dataset.get_raw_df(), self.args.Turbins,
                                               self.args.pred_len)
        show_metric_scores(dataset='valid', type='Separate', scores=sep_scores)
        sum_scores = all_turb_multi_interval_scores(preds, trues, self.val_dataset.get_raw_df(), self.args.Turbins,
                                                    self.args.pred_len)
        show_metric_scores(dataset='valid', type='Sum-up', scores=sum_scores)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            preds = []
            trues = []
            for i, batch in enumerate(self.train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred_10,pred_30,pred_60,true_10,true_30,true_60 = self._process_one_batch( batch )
                #print("pred shape:{}, true shape:{}".format(pred.shape,true.shape)) #[B,134,144]
                loss_10 = criterion(pred_10, true_10)
                # loss_30 = criterion(pred_30, true_30)
                # loss_60 = criterion(pred_60, true_60)
                # loss = loss_10+loss_30+loss_60
                loss = loss_10
                train_loss.append(loss.item())
                preds.append(pred_10.detach().cpu().numpy())
                trues.append(true_10.detach().cpu().numpy())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 梯度累积
                    loss = loss / self.args.acc_grad
                    loss.backward()

                    if ((i + 1) % self.args.acc_grad) == 0:
                        if self.args.max_grad_norm != 0:
                            # gradient clipping - this does it in place
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        model_optim.step()
                        model_optim.zero_grad()
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            # print('train shape:', preds.shape, trues.shape)  #
            # 逆标准化
            mean = np.expand_dims(self.train_dataset.data_mean[:, :, -1], 0)
            std = np.expand_dims(self.train_dataset.data_scale[:, :, -1], 0)
            # print("mean shape:{}, std shape:{}".format(mean.shape, std.shape))  # [1,134,1]
            preds = preds * std + mean
            trues = trues * std + mean

            preds = np.expand_dims(preds, -1).transpose([1, 0, 2, 3])
            trues = np.expand_dims(trues, -1).transpose([1, 0, 2, 3])
            print("train shape: ",preds.shape,trues.shape,self.train_dataset.get_raw_df()[0].shape)
            sep_scores = regressor_detailed_scores(preds, trues, self.train_dataset.get_raw_df(), self.args.Turbins,
                                                   self.args.pred_len)
            show_metric_scores(dataset='train', type='Separate', scores=sep_scores)
            sum_scores = all_turb_multi_interval_scores(preds, trues, self.train_dataset.get_raw_df(),
                                                        self.args.Turbins,
                                                        self.args.pred_len)
            show_metric_scores(dataset='train', type='Sum-up', scores=sum_scores)

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            torch.cuda.empty_cache()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.val_dataset, self.val_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        print("load best model: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path),strict=False)
        
        return self.model

    def test(self, setting):
        
        self.model.eval()
        
        preds = []
        trues = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                pred_10, pred_30, pred_60, true_10, true_30, true_60 = self._process_one_batch(batch)
                # print("pred shape:{}, true shape:{}".format(pred.shape,true.shape)) #[B,134,144]
                preds.append(pred_10.detach().cpu().numpy())
                trues.append(true_10.detach().cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        print('test shape:', preds.shape, trues.shape,self.test_dataset.get_raw_df()[0].shape) #
        # 逆标准化
        mean = np.expand_dims(self.train_dataset.data_mean[:, :, -1], 0)
        std = np.expand_dims(self.train_dataset.data_scale[:, :, -1], 0)
        print("mean shape:{}, std shape:{}".format(mean.shape,std.shape))# [1,134,1]
        preds = preds * std + mean
        trues = trues * std + mean

        preds = np.expand_dims(preds, -1).transpose([1, 0, 2, 3])
        trues = np.expand_dims(trues, -1).transpose([1, 0, 2, 3])

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        #print('mse:{}, mae:{}'.format(mse, mae))
        sep_scores = regressor_detailed_scores(preds, trues, self.test_dataset.get_raw_df(), self.args.Turbins,
                                               self.args.pred_len)
        show_metric_scores(dataset='test', type='Separate', scores=sep_scores)
        sum_scores = all_turb_multi_interval_scores(preds, trues, self.test_dataset.get_raw_df(), self.args.Turbins,
                                                    self.args.pred_len)
        show_metric_scores(dataset='test', type='Sum-up', scores=sum_scores)
        #np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path+'pred.npy', preds)
        #np.save(folder_path+'true.npy', trues)

        return


    def _process_one_batch(self, batch):

        batch_x_1, batch_sparse_x_1, batch_sparse_y_1, batch_y_1, \
        batch_x_3, batch_sparse_x_3, batch_sparse_y_3, batch_y_3, \
        batch_x_6, batch_sparse_x_6, batch_sparse_y_6, batch_y_6  = batch
        batch_x_1 = batch_x_1.float().to(self.device)
        batch_sparse_x_1 = batch_sparse_x_1.float().to(self.device)
        batch_sparse_y_1 = batch_sparse_y_1.float().to(self.device)
        batch_x_3 = batch_x_3.float().to(self.device)
        batch_sparse_x_3 = batch_sparse_x_3.float().to(self.device)
        batch_sparse_y_3 = batch_sparse_y_3.float().to(self.device)
        batch_x_6 = batch_x_6.float().to(self.device)
        batch_sparse_x_6 = batch_sparse_x_6.float().to(self.device)
        batch_sparse_y_6 = batch_sparse_y_6.float().to(self.device)

        batch_y_1 = batch_y_1.float()
        batch_y_3 = batch_y_3.float()
        batch_y_6 = batch_y_6.float()

        # decoder input
        #zy
        dec_inp_1 = torch.zeros([batch_y_1.shape[0], batch_y_1.shape[1], self.args.pred_len, batch_y_1.shape[-1]]).float().to(self.device)
        dec_inp_3 = torch.zeros([batch_y_3.shape[0], batch_y_3.shape[1], self.args.pred_len//3, batch_y_3.shape[-1]]).float().to(self.device)
        dec_inp_6 = torch.zeros([batch_y_6.shape[0], batch_y_6.shape[1], self.args.pred_len//6, batch_y_6.shape[-1]]).float().to(self.device)

        batch_sparse_y_1 = batch_sparse_y_1[:,:,-self.args.pred_len:,:]
        batch_sparse_y_3 = batch_sparse_y_3[:, :, -self.args.pred_len//3:, :]
        batch_sparse_y_6 = batch_sparse_y_6[:, :, -self.args.pred_len//6:, :]
        #print("dec_inp shape:{}".format(dec_inp.shape)) #[B, 134, 216, 12]
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_1, outputs_3, outputs_6 = self.model(
                        batch_x_1, batch_sparse_x_1, dec_inp_6, batch_sparse_y_6)[0]
                else:
                    outputs_1, outputs_3, outputs_6 = self.model(
                        batch_x_1, batch_sparse_x_1, dec_inp_6, batch_sparse_y_6)
        else:
            if self.args.output_attention:
                outputs_1, outputs_3, outputs_6 = self.model(
                    batch_x_1, batch_sparse_x_1, dec_inp_6, batch_sparse_y_6)[0]
            else:
                outputs_1, outputs_3, outputs_6 = self.model(
                    batch_x_1, batch_sparse_x_1, dec_inp_6, batch_sparse_y_6)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y_1 = batch_y_1[:, :, -self.args.pred_len:, f_dim:].to(self.device)
        batch_y_3 = batch_y_3[:, :, -self.args.pred_len//3:, f_dim:].to(self.device)
        batch_y_6 = batch_y_6[:, :, -self.args.pred_len//6:, f_dim:].to(self.device)


        return outputs_1,outputs_3,outputs_6, batch_y_1.squeeze(-1),batch_y_3.squeeze(-1),batch_y_6.squeeze(-1)
