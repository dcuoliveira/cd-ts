import pandas as pd
import numpy as np
from tqdm import tqdm
import pdblp # TODO - Install blpapi

import os
import time
import argparse

def parse_args():
    pass

def generate_dataset():
    con = pdblp.BCon(timeout=5000000)
    con.start()

    df = pd.read_excel("stock_indexes_metadata.xlsx")

    data = {}
    for s in tqdm(list(df["Region"].unique()),
                  total=len(list(df["Region"].unique()))):

        tickers_list = []
        for ticker in df.loc[df["Region"] == s]["BBG Ticker"].unique():
            # get data - PX_LAST means last price for a given day
            tmp_df = con.bdh(tickers=[ticker],
                             flds=["PX_LAST"],
                             start_date="20000101",
                             end_date="20220901")
            tmp_df.columns = tmp_df.columns.swaplevel(0, 1)
            tmp_df.columns = tmp_df.columns.droplevel()

            # resample data and forward fill for holidays
            tmp_df = tmp_df.resample("B").last().ffill()

            # append to tickers by region (s)
            tickers_list.append(tmp_df)

        # forward fill missings
        tickers_df = pd.concat(tickers_list, axis=1).ffill()

        # apply log difference transformation
        tickers_df = np.log(tickers_df).diff().dropna()

        data[s] = tickers_df

    data_all = np.array([data[s].to_numpy() for s in data.keys()])
    edges_all = None

    return data_all, edges_all

if __name__ == '__main__':
    savepath = os.path.dirname(__file__)
    suffix = "_stock_indices"
    data_all, edges_all = generate_dataset()

    np.save(
        os.path.join(savepath, "feat_train" + suffix + ".npy"),
        data_train,
    )