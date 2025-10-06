import pandas as pd
class TrainTestSplitter:
    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.df_train = None
        self.df_test = None

    def fit_transform(self, df):
        train_list = []
        test_list = []
        
        for name, group in df.groupby('names'):
            group = group.sort_values('date_mesure')
            n_train = int((1 - self.test_size) * len(group))
            train_list.append(group.iloc[:n_train])
            test_list.append(group.iloc[n_train:])
        
        self.df_train = pd.concat(train_list).reset_index(drop=True)
        self.df_test = pd.concat(test_list).reset_index(drop=True)
        return self.df_train, self.df_test

    def get_train(self):
        return self.df_train

    def get_test(self):
        return self.df_test
