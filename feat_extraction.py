import pandas as pd

from plotting import *
from preprocessing import *

if __name__ == "__main__":

    df_train = pd.read_csv("Data/sales_train.csv")
    df_test = pd.read_csv("Data/test.csv")

    # Residual information
    items = pd.read_csv("Data/items.csv")
    item_categories = pd.read_csv("Data/item_categories.csv")
    shops = pd.read_csv("Data/shops.csv")
    
    print(df_train.info())
    print(basic_details(df_train))