import logging
import sys

from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from typing import Dict
import warnings

from DataLoader.config import DATA_LOADER_PARAMS
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
handlers=[logging.FileHandler('logfile.log', encoding='utf-8'),
          logging.StreamHandler()])


class DataLoader:
    def __init__(self, params):
        warnings.filterwarnings("ignore")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.num_batches = params.get('num_batches', 13)
        self.raw_path = os.path.join(os.path.realpath(BASE_DIR), 'raw_data')
        self.types_path = os.path.join(os.path.realpath(BASE_DIR), 'types.pickle')
        self.datapath = os.path.join(os.path.realpath(BASE_DIR), 'datasets')
        self.year = params.get("year_to_split", 2010)
        self.step = 0

    def _extract(self, verbose=0):
        with open(self.types_path, 'rb') as f:
            dct = pickle.load(f)
            categ, orig_cols, orig_dtypes, svc_cols, svc_dtypes = dct['categ'], dct['orig cols'], dct['orig dtypes'], \
                                                                  dct['svc_cols'], dct['svc_dtypes']
        threshold = int((pd.to_datetime('2024-01-01') - pd.to_datetime(f'{self.year}-01-01')).days / 30)
        if self.num_batches > threshold:
            if verbose:
                logging.error("Process failed: batch_num is too large for this year_to_split")
            return None
        if verbose:
            logging.info('Reading static data...')
            logging.info('Note: it may take several time')
        orig_ = pd.read_csv(os.path.join(self.raw_path, 'historical_data_1999Q1.txt'), sep='|', names=orig_cols, dtype=orig_dtypes)
        orig_ = orig_.apply(lambda x: x.astype('category') if x.name in categ else x, axis=0)
        orig_ = orig_.reset_index(drop=True)
        if verbose:
            logging.info('Reading dynamic data...')
            logging.info('Note: it may take several time')
        svc_ = pd.read_csv(os.path.join(self.raw_path, 'historical_data_time1999Q1.csv'))
        svc_.period = pd.to_datetime(svc_.period.astype(str), format='%Y%m')

        min_date = pd.to_datetime(f'{self.year}-01-01')
        max_date = svc_['period'].max()

        delta = (max_date - min_date) // self.num_batches
        if verbose:
            logging.info('Creating train dataset...')
        tmp = svc_[(svc_.period < min_date)]
        tmp = tmp.sort_values(by=['id_loan', 'period'])
        tmp = tmp.reset_index(drop=True)
        tmp.zero_balance_code = tmp.zero_balance_code.astype('category')
        tmp = orig_.merge(
            tmp[['id_loan', 'zero_balance_code', 'period', 'zero_balance_effective_date']],
            on='id_loan', how='left')
        tmp = tmp.dropna(subset=['period'])
        tmp.to_csv(os.path.join(self.datapath, f"train_dataset.csv"))
        if verbose:
            logging.info('Creating batches...')
            p = tqdm(total=self.num_batches, file=sys.stdout, colour='WHITE')
        for i in range(self.num_batches):
            tmp = svc_[(svc_.period < min_date + (i+1) * delta) & (svc_.period >= min_date + (i) * delta)]
            tmp = tmp.sort_values(by=['id_loan', 'period'])
            tmp = tmp.reset_index(drop=True)
            tmp.zero_balance_code = tmp.zero_balance_code.astype('category')
            tmp = orig_.merge(
                tmp[['id_loan', 'zero_balance_code', 'period', 'zero_balance_effective_date']],
                on='id_loan', how='left')
            tmp = tmp.dropna(subset=['period'])
            tmp.to_csv(os.path.join(self.datapath, f"batch_{i}.csv"))
            if verbose: p.update(1)


    def create_datasets(self, verbose=0) -> None:

        """
        Creates training and finetuning datasets split by the specified year.
        Data is saved in CSV format to the `datasets` folder.

        -----------
        Returns:
            None

        The function creates files in the `datasets` folder:
            - `train_dataset.csv` — training dataset (all data before split_year).
            - `batch_i.csv` —  new datasets.
        """
        self.n = 0
        os.makedirs(self.datapath, exist_ok=True)
        self._extract(verbose=verbose)

    def _add_batch(self, f):
        if self.step < self.num_batches:
            self.step += 1
            df = pd.read_csv(os.path.join(self.datapath, r"train_dataset.csv"))
            if f:
                self.step -= 1
                logging.info('The batch was frozen, restoring previous data...')
            if self.step > 1:
                new = pd.read_csv(os.path.join(self.datapath, f"batch_{self.step}.csv"))
                min_ = new.period.min()
                new = pd.concat([df, new[df.columns]])
                new = self._create_time_parameter(new) # объединяли с train чтобы правильно рассчитать time
                new = new[new.period >= min_] # оставили только сам батч
            else:
                new = self._create_time_parameter(df)
            return new
        logging.error('No new batches left')
        return None

    def _create_time_parameter(self, df):
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

    def _first(self, f, verbose=0):
        if verbose==2: logging.info('Reading train data...')
        df = self._add_batch(f)
        dq = DataQualityEvaluator()
        if verbose == 2: logging.info('Evaluating data quality...')
        self.stats = dq.make_stats(df, self.step, verbose=(verbose >= 1))
        if verbose == 2: logging.info('Cleaning data...')
        df, counts = dq.fix_data(df, first_call=True)
        if verbose == 2:
            logging.info(f'{counts} bad column(s) was removed')
            logging.info(f'Done')
        return df

    def _step(self, f, verbose=0):
        if verbose == 2: logging.info('Reading next batch...')
        df_ = self._add_batch(f)
        if df_ is None: return None
        if verbose == 2: logging.info('Evaluating data quality...')
        dq = DataQualityEvaluator()
        result = dq.make_stats(df_, self.step, verbose=(verbose >= 1))
        if verbose == 2: logging.info('Cleaning data...')
        df_, counts = dq.fix_data(df_, first_call=False, stats=self.stats)
        # df_ = df_[list(set(df_.columns) & set(cols))]
        if verbose == 2:
            logging.info(f'{counts} bad column(s) was removed')
            logging.info(f'Done')
        self.stats[0] += result[0]
        self.stats[1] += result[1]
        return df_

    def get_data(self, verbose=0, freeze=False):
        if self.step == 0:
            return self._first(f=freeze, verbose=verbose)  # загрузка train датасета
        return self._step(f=freeze, verbose=verbose)


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

    def make_stats(self, df, numm, verbose=False):
        dct = [df.isna().sum(), df.shape[0]]
        f = open('stats.txt', 'w')
        c = self.completeness(df)
        f.write('completeness:\n')
        if verbose: print('completeness:')
        for k, v in c.items():
            f.write(f'    {k}: {v}\n')
            if verbose: print(f'    {k}: {v}')
        c = self.validity(df)
        f.write('\nvalidity:\n')
        if verbose: print('\nvalidity:')
        c = self.validity(df)
        for k, v in c.items():
            f.write(f'    {k:<25} - {v}\n')
            if verbose: print(f'    {k:<25} - {v}')
        f.write('\ntimeliness: True')
        if verbose: print('\ntimeliness: True')
        f.close()

        plt.figure(figsize=(20, 10))
        df.plot.box(vert=True, patch_artist=True, showmeans=True, grid=False)
        plt.xticks(rotation=45, ha='right',fontsize=8)
        plt.title(f"boxplot для батча {numm}")
        plt.tight_layout()
        plt.savefig(f'boxplot_batch_{numm}.png', dpi=150, bbox_inches='tight')
        cols = [['units_numb', 'occupancy_status'], ['loan_purpose', 'borrowers_num']]
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        for i in [0, 1]:
            for j in [0, 1]:
                sns.histplot(df[cols[i][j]], ax=ax[i][j], kde=True, color='skyblue', edgecolor='black')
                ax[i][j].set_title(f'Распределение {cols[i][j]} для батча {numm}', fontsize=8)

        plt.savefig(f'hists_batch_{numm}.png', dpi=150, bbox_inches='tight')

        df = self.create_feature(df)

        with open('stats.pickle', 'wb') as f:
            res = {}
            for col in df.columns[1:]:
                if df[col].dtype == 'object' and not (col in ['seller_name', 'service_name']):
                    res[col] = (dict(df['first_time_homebuyer_flag'].value_counts()), 'vc')
                if df[col].dtype == 'int':
                    res[col] = ([df[col].quantile(0.25), df[col].quantile(0.75)], 'q')
            pickle.dump(dct, f)
            pickle.dump(res, f)
        return dct

    def create_feature(self, df):
        # цена имущества = orig_UPB (кредит) * 100 / LTV
        # коэффициент долговой нагрузки = кредит / цена * DTI (соотношение долг/доход)
        df["debt_ratio"] = (df["LTV"]/100) * df["DTI_ratio"]
        # уровень риска, рассчитывается на основе DTI
        conditions = [
            (df["CLTV"] > 80) & (df["DTI_ratio"] > 45),
            (df["CLTV"] > 60) & (df["DTI_ratio"] > 35),
            (df["CLTV"] <= 60)
        ]
        choices = [1, 2, 3] # 1 - high risk, 2 - medium и 3-low
        df["risk_ratio"] = np.select(conditions, choices)
        return df

    def fix_data(self, a, first_call=False, stats=None):
        if not first_call:
            isnans = (a.isna().sum() + stats[0]) / (a.shape[0] + stats[1])
        else:
            isnans = a.isna().sum() / a.shape[0]
        # print(isnans)
        cols_num = len(a.columns)
        cols1 = a.isna().sum()[isnans < 0.4].index
        d = self.validity(a)
        cols2 = []
        for k, v in d.items():
            if not v: cols2.append(k)
        a = a[list(set(cols1) - set(cols2))]
        return a, cols_num - len(a.columns)


# data_loader = DataLoader(DATA_LOADER_PARAMS)
# df = data_loader.get_data(2)
# df = data_loader.get_data(2)
# df = data_loader.get_data(2, freeze=True)

# data_loader.create_datasets(1)
# for _ in range(DATA_LOADER_PARAMS['num_batch']+2):
#       df = data_loader.get_data(verbose=2)

