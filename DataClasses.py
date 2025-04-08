import pickle

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from zipfile import ZipFile
from typing import List
import re
import io
from typing import Dict
import warnings


class DataLoader:
    def __init__(self, path=None, year_to_split=2010, n_qtr=1):
        warnings.filterwarnings("ignore")
        self.raw_path = "raw_data" if path is None else path
        self.datapath = 'datasets'
        self.year = year_to_split
        self.n_qtr = n_qtr
        self.n = None
        self.step = 0

    def extract(self, n_qtr=None, start_date=None, end_date=None):
        patt = r"historical_data_\d{4}\.zip"
        patt2 = r"historical_data_(\d{4})\/historical_data_\1Q[1-4]\.zip"
        patt3 = r"historical_data_\d{4}Q[1-4]\.zip"
        with open('DataLoader/types.pickle', 'rb') as f:
            dct = pickle.load(f)
            categ, orig_cols, orig_dtypes, svc_cols, svc_dtypes = dct['categ'], dct['orig cols'], dct['orig dtypes'], \
                                                                  dct['svc_cols'], dct['svc_dtypes']
        k = 0
        res = pd.DataFrame()
        lst = os.listdir(self.raw_path)
        for elem in lst:  # по всем архивам
            if re.match(patt, elem):
                zp = ZipFile(os.path.join(self.raw_path, elem), 'r')
                for it in zp.namelist():  # по элементам в архиве (папка или Q-архивы)
                    if re.match(patt2, it) or re.match(patt3, it):
                        tmp = os.path.join(self.raw_path, elem, it.replace('/', '\\'))
                        zpQ = ZipFile(io.BytesIO(zp.open(it).read()), 'r')
                        zpQ_svc = zpQ.open(f'historical_data_time_{it[-10:-4]}.txt')
                        zpQ_orig = zpQ.open(f'historical_data_{it[-10:-4]}.txt')
                        print(k)
                        k += 1
                        svc_ = pd.read_csv(zpQ_svc, sep='|', names=svc_cols, dtype=svc_dtypes)
                        svc_.period = pd.to_datetime(svc_.period.astype(str), format='%Y%m')
                        if start_date:
                            svc_ = svc_[svc_.period >= pd.to_datetime(start_date)]
                        if end_date:
                            svc_ = svc_[svc_.period < pd.to_datetime(end_date)]
                        svc_ = svc_.sort_values(by=['id_loan', 'period'])
                        svc_ = svc_[(svc_.id_loan != svc_.id_loan.shift(1)) | (svc_.id_loan != svc_.id_loan.shift(-1))]
                        svc_ = svc_.reset_index(drop=True)
                        svc_.zero_balance_code = svc_.zero_balance_code.astype('category')
                        print(f'historical_data_time_{it[-10:-4]}.txt')
                        orig_ = pd.read_csv(zpQ_orig, sep='|', names=orig_cols, dtype=orig_dtypes)
                        orig_ = orig_.apply(lambda x: x.astype('category') if x.name in categ else x, axis=0)
                        orig_ = orig_.reset_index(drop=True)
                        print(f'historical_data_{it[-10:-4]}.txt')
                        tmp = orig_.merge(
                            svc_[['id_loan', 'zero_balance_code', 'period', 'zero_balance_effective_date']],
                            on='id_loan', how='left')
                        tmp = tmp.dropna(subset=['period'])
                        res = pd.concat([res, tmp])
                        if k == n_qtr:
                            res = res.sort_values(by=['id_loan', 'period'])
                            return res

        res['period'] = pd.to_datetime(res['period'].astype(str), format='%Y%m')
        res = res.sort_values(by=['id_loan', 'period'])
        return res

    def create_datasets(self, n_quaters: int = 1, split_year: int = 2010, path=None) -> None:

        """
        Creates training and finetuning datasets split by the specified year.
        Data is saved in CSV format to the `datasets` folder.

        Parameters:
        -----------
        n_quarters : int
            Number of quarters (3-month periods) to include in the dataset creation.
        split_year : int
            Year used as the split boundary:
            - All data before January 1 of `split_year` goes to the training set (train).
            - Data starting from January 1 of `split_year` is split into annual finetuning sets.

        Returns:
        -----------
        None
            The function creates files in the `datasets` folder:
            - `train_dataset.csv` — training dataset (all data before split_year).
            - `ft_dataset_{split_year+1}.csv`, `ft_dataset_{split_year+2}.csv`, etc. — annual finetuning datasets.
        """
        self.n = 0
        self.datapath = 'datasets' if path is None else path
        new_path = os.path.join(self.datapath, 'datasets')
        os.makedirs(new_path, exist_ok=True)

        end_date_ = f'{split_year}-01-01'
        train = self.extract(n_quaters, end_date=end_date_)
        train.to_csv(os.path.join(new_path, "train_dataset.csv"))

        for y in range(split_year + 1, 2024):
            self.n += 1
            start_date_ = f'{y - 1}-01-01'
            end_date_ = f'{y}-01-01'
            ft = self.extract(n_quaters, start_date=start_date_, end_date=end_date_)
            ft.to_csv(os.path.join(new_path, f"ft_dataset_{y}.csv"))

    def add_batch(self, df):
        if self.n is None:
            l = len(os.listdir('datasets'))
            self.n = l - 1 if l else 0
        if self.step < self.n:
            new = pd.read_csv(os.path.join(self.datapath, f"ft_dataset_{self.year + 1 + self.step}.csv"))
            new = pd.concat([df, new[df.columns]])
            self.step += 1
            return new
        return None

    def create_time_parameter(self, df):
        df.zero_balance_code.fillna(-1, inplace=True)
        df['time'] = None
        df['cens'] = (df.zero_balance_code != -1)
        df.sort_values(by=['id_loan', 'period'])
        tmp = pd.to_datetime((df[df.zero_balance_code.isin([-1])]).first_payment_date.astype(str), format='%Y%m')
        tmp = pd.to_datetime(df[df.zero_balance_code.isin([-1])].period) - tmp
        df.loc[df.zero_balance_code.isin([-1]), 'time'] = tmp
        df.loc[df.zero_balance_code.isin([-1]), 'time'] = pd.to_timedelta(
            df[df.zero_balance_code.isin([-1])].time, unit='D').dt.days // 30
        df = df[~(df.zero_balance_effective_date.isna() & df.cens)]
        df.loc[df.cens, 'time'] = pd.to_datetime(df[df.cens].zero_balance_effective_date.astype(str).str[:-2],
                                                 format='%Y%m') \
                                  - pd.to_datetime(df[df.cens].first_payment_date.astype(str), format='%Y%m')
        df.loc[df.cens, 'time'] = pd.to_timedelta(df[df.cens].time, unit='D').dt.days // 30
        # оставляем только одно (последнее) наблюдение с признаками первого
        # (из таблицы origination_data_file)
        start = (df.id_loan == df.shift(-1).id_loan)
        df = df[~start]
        return df[df.time >= 0]

    def first(self):
        return pd.read_csv('datasets/train_dataset.csv')

    def step_(self, df, verbose=0):
        if verbose==2: print('updating data...')
        df_ = self.add_batch(df)
        df_ = self.create_time_parameter(df_)
        if df_ is None: raise ValueError('step is out of range')
        if verbose==2: print('evaluating data quality...')
        dq = DataQualityEvaluator()
        result = dq.make_stats(df_, p=(verbose==1))
        if verbose==2: print('cleaning data...')
        df_, counts = dq.fix_data(df_)
        if verbose==2:
            print(f'{counts} bad column(s) was removed')
            print(f'\n updating completed')
        return df_


class DataQualityEvaluator:
    def __init__(self):
        pass

    def completeness(self, df) -> Dict[str, float]:
        df_bin = df.isna()
        cols_nan_ratio = df_bin.sum(axis=0) / df.shape[0]
        rows_nan_ratio = df_bin.sum(axis=1) / df.shape[1]
        d = {
            "full": df_bin.sum().sum() / np.prod(df.shape),
            "cols_max": cols_nan_ratio.max(),
            "rows_max": rows_nan_ratio.max()
        }
        return d

    def validity(self, df) -> Dict[str, float]:
        d = {}
        cols_to_check = ['credit_score', 'first_time_homebuyer_flag', 'MI_%',
                         'units_numb', 'occupancy_status', 'CLTV', 'DTI_ratio',
                         'LTV', 'channel', 'PPM_flag', 'amortization_type',
                         'property_type', 'loan_purpose', 'borrowers_num',
                         'program_ind', 'property_val_method', 'int_only_flag',
                         'MI_cancel_flag']
        arr = self.check_values(df)
        for i in range(len(cols_to_check)):  # проверка значений по документации для каждого столбца
            if not (arr[i] is None):
                d[cols_to_check[i]] = arr[i] == 0  # если True, то со столбцом все в порядке
        return d

    def check_values(self, a) -> List['float']:
        lst = []
        lst.append(a.credit_score[
                       (a.credit_score < 300) | (a.credit_score > 850)].sum() if 'credit_score' in a.columns else None)
        lst.append(a.first_time_homebuyer_flag[~a.first_time_homebuyer_flag.isin(
            ['N', 'Y', '9'])].sum() if 'first_time_homebuyer_flag' in a.columns else None)
        lst.append(
            a['MI_%'][((a['MI_%'] > 55) | (a['MI_%'] < 0)) & (a['MI_%'] != 999)].sum() if 'MI_%' in a.columns else None)
        lst.append(a.units_numb[~a.units_numb.isin([1, 2, 3, 4, 99])].sum() if 'units_numb' in a.columns else None)
        lst.append(a.occupancy_status[~a.occupancy_status.isin(
            ['P', 'I', 'S', '9'])].sum() if 'occupancy_status' in a.columns else None)
        lst.append(
            a.CLTV[~(((a.CLTV >= 6) & (a.CLTV <= 200)) | (a.CLTV == 999))].sum() if 'CLTV' in a.columns else None)
        lst.append(a.DTI_ratio[~(((a.DTI_ratio >= 0) & (a.DTI_ratio <= 65)) | (
                a.DTI_ratio == 999))].sum() if 'DTI_ratio' in a.columns else None)
        lst.append(a.LTV[~(((a.LTV >= 6) & (a.LTV <= 105)) | (a.LTV == 999))].sum() if 'LTV' in a.columns else None)
        lst.append(a.channel[~a.channel.isin(['R', 'B', 'C', 'T', '9'])].sum() if 'channel' in a.columns else None)
        lst.append(a.PPM_flag[~a.PPM_flag.isin(['Y', 'N'])].sum() if 'PPM_flag' in a.columns else None)
        lst.append(a.amortization_type[
                       ~a.amortization_type.isin(['FRM', 'ARM'])].sum() if 'amortization_type' in a.columns else None)
        lst.append(a.property_type[~a.property_type.isin(
            ['CO', 'PU', 'MH', 'SF', 'CP', '99'])].sum() if 'property_type' in a.columns else None)
        lst.append(a.loan_purpose[
                       ~a.loan_purpose.isin(['P', 'C', 'N', 'R', '9'])].sum() if 'loan_purpose' in a.columns else None)
        lst.append(a.borrowers_num[~a.borrowers_num.isin([1, 2, 99])].sum() if 'borrowers_num' in a.columns else None)
        lst.append(
            a.program_ind[~a.program_ind.isin(['H', 'F', 'R', '9', 9])].sum() if 'program_ind' in a.columns else None)
        lst.append(a.property_val_method[~a.property_val_method.isin(
            [1, 2, 3, 4, 9])].sum() if 'property_val_method' in a.columns else None)
        lst.append(a.int_only_flag[~a.int_only_flag.isin(['Y', 'N'])].sum() if 'int_only_flag' in a.columns else None)
        lst.append(a.MI_cancel_flag[~a.MI_cancel_flag.isin(
            ['Y', 'N', 7, 9, '7', '9'])].sum() if 'MI_cancel_flag' in a.columns else None)
        return lst

    def make_stats(self, df, p=False):
        dct = {}
        f = open('DataLoader/stats.txt', 'w')
        c = self.completeness(df)
        dct['completeness'] = c
        f.write('completeness:\n')
        if p: print('completeness:')
        for k, v in c.items():
            f.write(f'    {k}: {v}\n')
            if p: print(f'    {k}: {v}')
        c = self.validity(df)
        dct['validity'] = c
        f.write('\nvalidity:\n')
        if p: print('\nvalidity:')
        c = self.validity(df)
        for k, v in c.items():
            f.write(f'    {k:<25} - {v}\n')
            if p: print(f'    {k:<25} - {v}')
        dct['timeliness'] = {'timeliness': True}
        f.write('\ntimeliness: True')
        if p: print('\ntimeliness: True')
        f.close()
        with open('stats.pickle', 'wb') as f:
            pickle.dump(dct, f)
        return dct

    def fix_data(self, a):
        cols_num = len(a.columns)
        cols1 = a.isna().sum()[(a.isna().sum() / a.shape[0]) < 0.4].index
        d = self.validity(a)
        cols2 = []
        for k, v in d.items():
            if not v: cols2.append(k)
        a = a[list(set(cols1) - set(cols2))]
        return a, cols_num-len(a.columns)


# data_loader = DataLoader()
# df1 = data_loader.first()
# df1 = data_loader.stepp(df1, verbose=2)