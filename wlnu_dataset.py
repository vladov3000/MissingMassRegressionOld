import pandas as pd
import torch

class WLNuDataset(torch.utils.data.Dataset):
    def __init__(self, df_path="data/gen.csv"):
        self.df = pd.read_csv(df_path)
        print(self.df)

