import pandas as pd

class Data:
    @staticmethod
    def get_dfs():
        df_nomal= pd.read_csv('../data/normal/IDS.csv')
        df_X=df_nomal[['Id', 'Season', 'Episode', 'Chunk']]
        df_Y=df_nomal[['Id', 'Ex', 'Ey', 'Px', 'Py']]
        return df_nomal,df_X, df_Y
