import pandas
import uproot


def root_to_pandas(
    file_name: str,
    columns: list[str],
    tree_name: str = 'outTree'
) -> pandas.DataFrame:
    '''
    Convert a ROOT TTree into a pandas DataFrame.
    
    Arguments:
    - file_name (str): name of the ROOT file to be converted.
    - columns (list[str]): list of feature names to be imported.
    - tree_name (str): name of the TTree object.
    '''
    with uproot.open(file_name) as f:
        df = f[tree_name].arrays(columns, library='pd')
    return df


def train_test_sample(
    df: pandas.DataFrame,
    n_samples: int,
    frac_train: float = 0.5,
    weight_name: str = 'w',
    random_state: int = None
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    '''
    Split and sample a pandas DataFrame into train and test datasets.

    Arguments:
    df (pandas.DataFrame): input dataframe.
    n_samples (int): total number of events to be sampled.
    frac_train (float): fractional size of train sample.
    weight_name (str): name of the weight to be used during sampling.
    '''
    # Split train and test datasets
    df_train = df.sample(frac=frac_train, replace=False, random_state=random_state)
    df_test = df.drop(df_train.index)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    # Sample train and test datasets according to weight
    n_train = int(n_samples*frac_train)
    n_test = int(n_samples*(1-frac_train))
    df_train = df_train.sample(n=n_train, replace=True, weights=df_train.loc[:, weight_name], random_state=random_state)
    df_test = df_test.sample(n=n_test, replace=True, weights=df_test.loc[:, weight_name], random_state=random_state)
    # Remove weight column as no longer needed
    df_train.drop(['w'], axis=1, inplace=True)
    df_test.drop(['w'], axis=1, inplace=True)
    return df_train, df_test
