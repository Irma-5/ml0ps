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

path = "D:\SurvivalAnalysis\CompetingRisk\LoanData" # тут написать свое
patt = r"historical_data_\d{4}\.zip"
patt2 = r"historical_data_(\d{4})\/historical_data_\1Q[1-4]\.zip"
patt3 = r"historical_data_\d{4}Q[1-4]\.zip"
svc_cols = ['id_loan', 'period', 'curr_act_UPB', 'delinq_status',
                'loan_age', 'remaining_months', 'deffect_settlement_date',
                'modification_flag', 'zero_balance_code', 'zero_balance_effective_date',
                'curr_interest_rate', 'curr_deferred_UPB', 'DDLPI', 'MI_recoveries',
                'net_sales_proceeds', 'non_MI_recoveries', 'expenses', 'legal_costs',
                'maint_preserv_costs', 'taxs_insur_costs', 'miscell_expenses',
                'actual_loss', 'mod_cost', 'step_mod_flag', 'deferred_payment_plan',
                'ELTV', 'zero_balance_rem_UPB', 'delinq_accur_interest',
                'delinq_due_disaster', 'borrowe_asistance_stat_code', 'curr_month_mod_cost',
                'interest_bearing_UPB']
orig_cols = ['credit_score', 'first_payment_date', 'first_time_homebuyer_flag',
                      'maturity_date', 'MSA', 'MI_%', 'units_numb', 'occupancy_status',
                      'CLTV', 'DTI_ratio', 'orig_UPB', 'LTV', 'orig_interest_rate',
                      'channel', 'PPM_flag', 'amortization_type',
                      'property_state', 'property_type', 'postal_code', 'id_loan',
                      'loan_purpose', 'orig_loan_term', 'borrowers_num', 'seller_name',
                      'service_name', 'super_conf_flag', 'id_loan_preharp',
                      'program_ind', 'HARP_ind', 'property_val_method',
                      'int_only_flag', 'MI_cancel_flag']

svc_dtypes = {'zero_balance_code': 'object', 'id_loan': 'object', 'period': 'str',
             'zero_balance_effective_date': 'str'}

orig_dtypes = {'credit_score': 'Int16', 'first_payment_date': 'str', 'first_time_homebuyer_flag': 'str',
              'maturity_date': 'str', 'MSA': 'Int32', 'MI_%': 'Int16', 'units_numb': 'Int8', 'occupancy_status': 'str',
              'CLTV': 'Int16', 'DTI_ratio': 'Int16', 'orig_UPB': 'Int64', 'LTV': 'Int16', 'orig_interest_rate': 'str',
              'channel': 'str', 'PPM_flag': 'str', 'amortization_type': 'str',
              'property_state': 'str', 'property_type': 'str', 'postal_code': 'Int32', 'id_loan': 'str',
              'loan_purpose': 'str', 'orig_loan_term': 'Int16', 'borrowers_num': 'Int8', 'seller_name': 'str',
              'service_name': 'str', 'super_conf_flag': 'str', 'id_loan_preharp': 'str',
              'program_ind': 'str', 'HARP_ind': 'str', 'property_val_method': 'Int64',
                      'int_only_flag': 'str', 'MI_cancel_flag': 'str'}

class DataLoader:
    def __init__(self, path, year_to_split=2010, n_qtr=1):
        self.datapath = "D:\SurvivalAnalysis\CompetingRisk" if path is None else path
        self.year = year_to_split
        self.n_qtr = n_qtr
        self.n = 0
        self.curr = 0

    def extract(self, n_qtr=None, start_date=None, end_date=None):
        k = 0
        res = pd.DataFrame()
        lst = os.listdir(path)
        for elem in lst:  # по всем архивам
            if re.match(patt, elem):
                zp = ZipFile(os.path.join(path, elem), 'r')
                for it in zp.namelist():  # по элементам в архиве (папка или Q-архивы)
                    if re.match(patt2, it) or re.match(patt3, it):
                        tmp = os.path.join(path, elem, it.replace('/', '\\'))
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

    def create_datasets(self, n_quaters: int = 1, split_year: int = 2010, path=datapath) -> None:

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

        new_path = os.path.join(path, 'datasets')
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

    def step(self, df):
        if self.curr < self.n:
            new = pd.read_csv(os.path.join(self.datapath, f"ft_dataset_{self.year + 1 + self.curr}.csv"))
            new = pd.concat(df, new)
            return new
        return None


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
            d[cols_to_check[i]] = arr[i] == 0  # если True, то со столбцом все в порядке
        return d

    def check_values(self, a) -> List['float']:
        return [a.credit_score[(a.credit_score < 300) | (a.credit_score > 850)].sum(),
                a.first_time_homebuyer_flag[~a.first_time_homebuyer_flag.isin(['N', 'Y', '9'])].sum(),
                a['MI_%'][((a['MI_%'] > 55) | (a['MI_%'] < 0)) & (a['MI_%'] != 999)].sum(),
                a.units_numb[~a.units_numb.isin([1, 2, 3, 4, 99])].sum(),
                a.occupancy_status[~a.occupancy_status.isin(['P', 'I', 'S', '9'])].sum(),
                a.CLTV[~(((a.CLTV >= 6) & (a.CLTV <= 200)) | (a.CLTV == 999))].sum(),
                a.DTI_ratio[~(((a.DTI_ratio >= 0) & (a.DTI_ratio <= 65)) | (a.DTI_ratio == 999))].sum(),
                a.LTV[~(((a.LTV >= 6) & (a.LTV <= 105)) | (a.LTV == 999))].sum(),
                a.channel[~a.channel.isin(['R', 'B', 'C', 'T', '9'])].sum(),
                a.PPM_flag[~a.PPM_flag.isin(['Y', 'N'])].sum(),
                a.amortization_type[~a.amortization_type.isin(['FRM', 'ARM'])].sum(),
                a.property_type[~a.property_type.isin(['CO', 'PU', 'MH', 'SF', 'CP', '99'])].sum(),
                a.loan_purpose[~a.loan_purpose.isin(['P', 'C', 'N', 'R', '9'])].sum(),
                a.borrowers_num[~a.borrowers_num.isin([1, 2, 99])].sum(),
                a.program_ind[~a.program_ind.isin(['H', 'F', 'R', '9', 9])].sum(),
                a.property_val_method[~a.property_val_method.isin([1, 2, 3, 4, 9])].sum(),
                a.int_only_flag[~a.int_only_flag.isin(['Y', 'N'])].sum(),
                a.MI_cancel_flag[~a.MI_cancel_flag.isin(['Y', 'N', 7, 9, '7', '9'])].sum()]

    def print_params(self, df):
        print('completeness:')
        c = self.completeness(df)
        for k, v in c.items():
            print(f'    {k}: {v}')
        print('\nvalidity:')
        c = self.validity(df)
        for k, v in c.items():
            print(f'    {k:<25} - {v}')
        print('\ntimeliness: True')

    def fix_data(self, a):
        cols1 = a.isna().sum()[(a.isna().sum() / a.shape[0]) < 0.4].index
        d = self.validity(a)
        cols2 = []
        for k, v in d.items():
            if v: cols2.append(k)
        return a[list(set(cols1) & set(cols2))]