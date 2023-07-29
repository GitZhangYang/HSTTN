import numpy as np
import pandas as pd


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe


def ignore_zeros(predictions, grounds):
    """
    Desc:
        Ignore the zero values for evaluation
    Args:
        predictions:
        grounds:
    Returns:
        Predictions and ground truths
    """
    preds = predictions[np.where(grounds != 0)]
    gts = grounds[np.where(grounds != 0)]
    return preds, gts


def rse(pred, ground_truth):
    """
    Desc:
        Root square error
    Args:
        pred:
        ground_truth: ground truth vector
    Returns:
        RSE value
    """
    _rse = 0.
    if len(pred) > 0 and len(ground_truth) > 0:
        _rse = np.sqrt(np.sum((ground_truth - pred)**2)) / np.sqrt(
            np.sum((ground_truth - ground_truth.mean())**2))
    return _rse


def corr(pred, gt):
    """
    Desc:
        Correlation between the prediction and ground truth
    Args:
        pred:
        gt: ground truth vector
    Returns:
        Correlation
    """
    _corr = 0.
    if len(pred) > 0 and len(gt) > 0:
        u = ((gt - gt.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = np.sqrt(((gt - gt.mean(0))**2 * (pred - pred.mean(0))**2).sum(0))
        _corr = (u / d).mean(-1)
    return _corr


def mae(pred, gt):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAE value
    """
    _mae = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSE value
    """
    _mse = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt)**2)
    return _mse


def rmse(pred, gt):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        RMSE value
    """
    return np.sqrt(mse(pred, gt))


def mape(pred, gt):
    """
    Desc:
        Mean Absolute Percentage Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAPE value
    """
    _mape = 0.
    if len(pred) > 0 and len(gt) > 0:
        #_mape = np.mean(np.abs((pred - gt) / (gt)))
        _mape = np.sum(np.abs(pred-gt))/np.sum(np.abs(gt))
    #if _mape>100:
        #print("wrong pred:{}, true:{}".format(pred,gt))
    return _mape

def medape(pred, gt):
    _medape = 0.
    if len(pred) > 0 and len(gt) > 0:
        _medape = np.median(np.abs((pred - gt) / (gt)))
    return _medape

def smape(pred, gt):
    _smape = 0.
    if len(pred) > 0 and len(gt) > 0:
        _smape = np.mean(2.0 * np.abs(pred - gt) / (np.abs(pred) + np.abs(gt)))
    return _smape

def mspe(pred, gt):
    """
    Desc:
        Mean Square Percentage Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSPE value
    """
    return np.mean(np.square((pred - gt) / gt)) if len(pred) > 0 and len(
        gt) > 0 else 0


def regressor_scores(prediction, gt):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        prediction:
        gt: ground truth vector
    Returns:
        A tuple of metrics
    """
    _mae = mae(prediction, gt)
    _rmse = rmse(prediction, gt)
    _mape = mape(prediction, gt)
    #_medape = medape(prediction, gt)
    _smape = smape(prediction, gt)
    return _mae, _rmse, _mape, _smape


def turbine_scores(pred, gt, cond, begin, end, stride=1):
    """
    Desc:
        Calculate the average MAE and RMSE of one turbine
    Args:
        pred: prediction for one turbine
        gt: ground truth
        raw_data: the DataFrame of one wind turbine
        begin, end: 当前时间区间
        stride:
    Returns:
        The averaged MAE and RMSE
    """
    maes, rmses, mapes, smapes = [], [], [], []
    cnt_sample, out_seq_len, _ = pred.shape

    for i in range(0, cnt_sample, stride):
        _cond = cond[i:out_seq_len + i] #当前样本的valid情况
        indices = np.where(_cond[begin:end]) #当前时间区间
        prediction = pred[i,begin:end,:]
        prediction = prediction[indices]
        targets = gt[i,begin:end,:]
        targets = targets[indices]
        _mae, _rmse, _mape, _smape = regressor_scores(prediction[:] / 1000, targets[:] / 1000)
        if _mae != _mae or _rmse != _rmse:
            continue
        maes.append(_mae)
        rmses.append(_rmse)
        mapes.append(_mape)
        smapes.append(_smape)
    avg_mae = np.array(maes).mean()
    avg_rmse = np.array(rmses).mean()
    avg_mape = np.array(mapes).mean()
    avg_smape = np.array(smapes).mean()
    return avg_mae, avg_rmse, avg_mape, avg_smape

def regressor_detailed_scores(predictions, gts, raw_df_lst, capacity,
                              output_len):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        predictions:
        gts: ground truth vector
        raw_df_lst:
        settings:
    Returns:
        A tuple of metrics
    """
    #all_mae, all_rmse ,all_mape = [], [], []
    maes_full, rmses_full, mapes_full, smapes_full = [], [], [], []
    maes_2_div, rmses_2_div, mapes_2_div, smapes_2_div = [], [], [], []
    for i in range(capacity):
        prediction = predictions[i]
        gt = gts[i]
        raw_data = raw_df_lst[i]
        nan_cond = pd.isna(raw_data).any(axis=1)
        invalid_cond = (raw_data['Patv'] < 0) | \
                       ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                       ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                       ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                        (raw_data['Ndir'] > 720))
        cond = (~invalid_cond) & (~nan_cond)
        _mae, _rmse, _mape, _smape = turbine_scores(prediction, gt, cond, 0, output_len, 1)
        maes_full.append(_mae)
        rmses_full.append(_rmse)
        mapes_full.append(_mape)
        smapes_full.append(_smape)

        # 划分2个时间段
        mae_tmp, rmse_tmp, mape_tmp, smape_tmp = [], [], [], []
        for i in range(2):
            _mae, _rmse, _mape, _smape = turbine_scores(prediction, gt, cond, i * 72, (i + 1) * 72, 1)
            mae_tmp.append(_mae)
            rmse_tmp.append(_rmse)
            mape_tmp.append(_mape)
            smape_tmp.append(_smape)
        maes_2_div.append(mae_tmp[:])
        rmses_2_div.append(rmse_tmp[:])
        mapes_2_div.append(mape_tmp[:])
        smapes_2_div.append(smape_tmp[:])

    print('maes_full shape:{}'.format(np.array(maes_full).shape))
    total_mae_full = np.array(maes_full).sum()
    total_rmse_full = np.array(rmses_full).sum()
    total_mape_full = np.array(mapes_full).sum()
    total_smape_full = np.array(smapes_full).sum()
    total_mae_2 = np.array(maes_2_div).sum(axis=0)
    total_rmse_2 = np.array(rmses_2_div).sum(axis=0)
    total_mape_2 = np.array(mapes_2_div).sum(axis=0)
    total_smape_2 = np.array(smapes_2_div).sum(axis=0)

    return total_mae_full, total_rmse_full, total_mape_full, total_smape_full, \
            total_mae_2, total_rmse_2, total_mape_2, total_smape_2


def turbine_invalid_filter(pred, gt, raw_data, examine_len, stride=1):

    nan_cond = pd.isna(raw_data).any(axis=1)

    invalid_cond = (raw_data['Patv'] < 0) | \
                   ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                   ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                   ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                    (raw_data['Ndir'] > 720))

    cond = invalid_cond | nan_cond

    preds_ls  = []
    trues_ls = []
    cnt_sample, out_seq_len, _ = pred.shape
    for i in range(0, cnt_sample, stride):
        indices = np.where(cond[i:out_seq_len + i])
        prediction = pred[i]
        prediction[indices] = 0
        preds_ls.append(prediction)
        targets = gt[i]
        targets[indices] = 0
        trues_ls.append(targets)

    return np.array(preds_ls), np.array(trues_ls)

def all_turb_multi_interval_scores(predictions, gts, raw_df_lst, capacity,output_len):
    """
    predictions, trues: [turb,samp,outlen,1]
    raw_df:[turb,samp*,attribute]
    """
    print('preds shape:{}'.format(predictions.shape))
    print('trues shape:{}'.format(gts.shape))
    #将pred和true的invalid cond置零，然后将全部turb的相加，避免mape除0的情况
    preds = []
    trues = []
    for i in range(capacity):
        prediction = predictions[i]
        gt = gts[i]
        raw_df = raw_df_lst[i]
        turb_preds, turb_trues = turbine_invalid_filter(prediction, gt, raw_df, output_len, 1)
        preds.append(turb_preds)
        trues.append(turb_trues)
    preds = np.array(preds)
    trues = np.array(trues)
    preds = np.sum(preds,axis=0)
    trues = np.sum(trues,axis=0)
    print('sum preds shape:{}'.format(preds.shape))
    print('sum trues shape:{}'.format(trues.shape))
    maes_full, rmses_full, mapes_full, smapes_full = [], [], [], []
    maes_2_div, rmses_2_div, mapes_2_div, smapes_2_div = [], [], [], []
    cnt_sample, out_seq_len, _ = preds.shape
    for i in range(0, cnt_sample):
        prediction = preds[i]
        targets = trues[i]
        # 整个时间段
        mae_full, rmse_full, mape_full, smape_full = regressor_scores(prediction[:] / 1000, targets[:] / 1000)
        maes_full.append(mae_full)
        rmses_full.append(rmse_full)
        mapes_full.append(mape_full)
        smapes_full.append(smape_full)
        # 划分为2个时间段
        mae_tmp, rmse_tmp, mape_tmp, smape_tmp = [], [], [], []
        for i in range(2):
            mae_2, rmse_2, mape_2, smape_2 = regressor_scores(prediction[i * 72:(i + 1) * 72] / 1000,
                                                     targets[i * 72:(i + 1) * 72] / 1000)
            mae_tmp.append(mae_2)
            rmse_tmp.append(rmse_2)
            mape_tmp.append(mape_2)
            smape_tmp.append(smape_2)
        maes_2_div.append(mae_tmp[:])
        rmses_2_div.append(rmse_tmp[:])
        mapes_2_div.append(mape_tmp[:])
        smapes_2_div.append(smape_tmp[:])

    print('mae_full shape:{}'.format(np.array(maes_full).shape))
    avg_mae_full = np.array(maes_full).mean()
    avg_rmse_full = np.array(rmses_full).mean()
    avg_mape_full = np.array(mapes_full).mean()
    avg_smape_full = np.array(smapes_full).mean()
    avg_mae_2 = np.mean(np.array(maes_2_div),axis=0)
    avg_rmse_2 = np.mean(np.array(rmses_2_div),axis=0)
    avg_mape_2 = np.mean(np.array(mapes_2_div),axis=0)
    avg_smape_2 = np.mean(np.array(smapes_2_div), axis=0)

    return avg_mae_full, avg_rmse_full, avg_mape_full, avg_smape_full,\
            avg_mae_2, avg_rmse_2, avg_mape_2, avg_smape_2

def regressor_metrics(pred, gt):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        pred:
        gt: ground truth vector
    Returns:
        A tuple of metrics
    """
    _mae = mae(pred, gt)
    _mse = mse(pred, gt)
    _rmse = rmse(pred, gt)
    # pred, gt = ignore_zeros(pred, gt)
    _mape = mape(pred, gt)
    _mspe = mspe(pred, gt)
    return _mae, _mse, _rmse, _mape, _mspe

def show_metric_scores(dataset, type, scores):
    total_mae_full, total_rmse_full, total_mape_full, total_smape_full, \
    total_mae_2, total_rmse_2, total_mape_2, total_smape_2 = scores
    print("Turbine {} Scores".format(type))
    print("{} full MAE:{}, RMSE:{}, MAPE:{}, SMAPE:{}".format(dataset, total_mae_full, total_rmse_full, total_mape_full,
                                                                                total_smape_full))
    print("{} MAE 2 intervals: ".format(dataset), total_mae_2)
    print("{} RMSE 2 intervals: ".format(dataset), total_rmse_2)
    print("{} MAPE 2 intervals: ".format(dataset), total_mape_2)
    print("{} SMAPE 2 intervals: ".format(dataset), total_smape_2)
