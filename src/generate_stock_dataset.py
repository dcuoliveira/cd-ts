import pandas as pd
from tqdm import tqdm
import numpy as np
import os

try:
    import pdblp
    con = pdblp.BCon(timeout=50000)
    con.start()
except:
    raise Exception("You do not have a bloomberg terminal installed, please ask for the dataset with the authors")

DEBUG = True

def generate_stock_dataset(savepath):
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "data", "utils", "stock_index_metadata.xlsx"))

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

        # save elements of the dataset
        np.savez(file=os.path.join(savepath, s + suffix + ".npz"),
                 X_np=tickers_df.to_numpy(),
                 Gref=None,
                 n=np.array(tickers_df.shape[1]),
                 T=np.array(tickers_df.shape[0]))

if __name__ == '__main__':
    if DEBUG:
        savepath = os.path(os.path.dirname(__file__), "data/world_stock_indexes")
        suffix = "_stock_indexes"

        generate_stock_dataset(savepath=savepath,
                               suffix=suffix)