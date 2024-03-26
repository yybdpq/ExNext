import os
import pickle
import pandas as pd
from datetime import datetime
import os.path as osp
from util.cfg import Cfg
from util.funcs import pathget
from pre.pre_funcs import dropfirst, keeplast, encodeID, remain_available
from pre.generate_file import GenerateFile
from pre.generate_hypergraph import generate_hypergf
import logging

#TKY&CA
def pre_tkyca(cfg: Cfg, path: bytes) -> pd.DataFrame:
    if cfg.dataset_args.dataset_name == 'tky':
        raw_file = 'dataset_TSMC2014_TKY.txt'
    else:
        raw_file = 'dataset_gowalla_ca_ne.csv'

    GenerateFile.root_path = path
    data = GenerateFile.read_data(raw_file, cfg.dataset_args.dataset_name)
    data = GenerateFile.filter_low_freq(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
    data = GenerateFile.split_data(data)
    
    #for ca dataset, after one round of filter, there still be many low frequency pois and users, so do twice
    if cfg.dataset_args.dataset_name == 'ca':
        data = GenerateFile.filter_low_freq(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = GenerateFile.split_data(data)

    data = GenerateFile.generateID(
        data,
        cfg.dataset_args.session_time_interval,
        cfg.dataset_args.do_label_encode,
        cfg.dataset_args.only_last_metric
    )
    return data

#NYC
def pre_nyc(path: bytes, preprocessed_path: bytes) -> pd.DataFrame:
    raw_path = osp.join(path, 'raw')

    df_train = pd.read_csv(osp.join(raw_path, 'NYC_train.csv'))
    df_val = pd.read_csv(osp.join(raw_path, 'NYC_val.csv'))
    df_test = pd.read_csv(osp.join(raw_path, 'NYC_test.csv'))
    df_train['SplitTag'] = 'train'
    df_val['SplitTag'] = 'validation'
    df_test['SplitTag'] = 'test'
    df = pd.concat([df_train, df_val, df_test])
    df.columns = [
        'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryCode', 'PoiCategoryName', 'Latitude', 'Longitude',
        'TimezoneOffset', 'UTCTime', 'UTCTimeOffset', 'UTCTimeOffsetWeekday', 'UTCTimeOffsetNormInDayTime',
        'pseudo_session_trajectory_id', 'UTCTimeOffsetNormDayShift', 'UTCTimeOffsetNormRelativeTime', 'SplitTag'
    ]
    
    #data transformation
    df['trajectory_id'] = df['pseudo_session_trajectory_id']
    df['UTCTimeOffset'] = df['UTCTimeOffset'].apply(lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))
    df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%s'))
    df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
    df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
    df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
    df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)
    
    #encoding ID
    df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
    traj_id_le, padding_traj_id = encodeID(df, df, 'pseudo_session_trajectory_id')
    
    df_train = df[df['SplitTag'] == 'train']
    poi_id_le, padding_poi_id = encodeID(df_train, df, 'PoiId')
    poi_category_le, padding_poi_category = encodeID(df_train, df, 'PoiCategoryId')
    user_id_le, padding_user_id = encodeID(df_train, df, 'UserId')
    hour_id_le, padding_hour_id = encodeID(df_train, df, 'UTCTimeOffsetHour')
    weekday_id_le, padding_weekday_id = encodeID(df_train, df, 'UTCTimeOffsetWeekday')

    with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
        pickle.dump([
            poi_id_le, poi_category_le, user_id_le, hour_id_le, weekday_id_le,
            padding_poi_id, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id
        ], f)

    #ignore the first for train/validate/test and keep the last for validata/test
    df = dropfirst(df)
    df = keeplast(df)
    return df

#main pre
def preprocess(cfg: Cfg):
    root_path = pathget()
    dataset_name = cfg.dataset_args.dataset_name
    data_path = osp.join(root_path, 'data', dataset_name)
    preprocessed_path = osp.join(data_path, 'preprocessed')
    sample_file = osp.join(preprocessed_path, 'sample.csv')
    train_file = osp.join(preprocessed_path, 'train_sample.csv')
    validate_file = osp.join(preprocessed_path, 'validate_sample.csv')
    test_file = osp.join(preprocessed_path, 'test_sample.csv')

    keep_cols = [
        'check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
        'query_pseudo_session_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
        'PoiCategoryName', 'last_checkin_epoch_time'
    ]

    if not osp.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    
    #data preprocess
    if not osp.exists(sample_file):
        if 'nyc' == dataset_name:
            keep_cols += ['trajectory_id']
            preprocessed_data = pre_nyc(data_path, preprocessed_path)
        elif 'tky' == dataset_name :
            preprocessed_data = pre_tkyca(cfg, data_path)
        elif 'ca' == dataset_name:
            preprocessed_data = pre_tkyca(cfg, data_path)
        else:
            raise ValueError(f'Wrong dataset name: {dataset_name} ')
        
        preprocessed_result = remain_available(preprocessed_data)
        preprocessed_result['sample'].to_csv(sample_file, index=False)
        preprocessed_result['train_sample'][keep_cols].to_csv(train_file, index=False)
        preprocessed_result['validate_sample'][keep_cols].to_csv(validate_file, index=False)
        preprocessed_result['test_sample'][keep_cols].to_csv(test_file, index=False)

    #generate hypergraph
    if not osp.exists(osp.join(preprocessed_path, 'ci2traj_pyg_data.pt')):
        generate_hypergf(sample_file, preprocessed_path, cfg.dataset_args)

    logging.info('[Preprocess] Done preprocessing.')
    print("[Preprocess] Done preprocessing.")
