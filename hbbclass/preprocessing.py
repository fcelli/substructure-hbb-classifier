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

def train_test_split(
    df: pandas.DataFrame,
    frac_train: float = 0.5
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    '''
    Split a pandas DataFrame into train and test samples.
    
    Arguments:
    df (pandas.DataFrame): input dataframe.
    frac_train (float): fractional size of train sample.
    '''
    df_train = df.sample(frac=frac_train)
    df_test = df.drop(df_train.index)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_test