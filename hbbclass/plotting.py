import matplotlib
import matplotlib.pyplot as plt
import pandas


def compare_feature(
    df: pandas.DataFrame,
    feature_name: str,
    limits: tuple[float, float] = None,
    bins: int = 10,
    x_label: str = '',
    y_label: str = '',
    ax: matplotlib.axes = None
) -> None:
    '''
    Compare the signal and background distributions of a feature.
    
    Arguments:
    - df (pandas.DataFrame): input dataframe containing the data to be plotted.
    - feature_name (str): name of the feature to be plotted
    - limits (tuple[float, float]): lower and upper limits of the histogram.
    - bins (int): number of bins of the histogram.
    '''
    if limits is not None:
        df = df.loc[(df[feature_name] >= limits[0]) & (df[feature_name] <= limits[1])]
    df_sig = df.loc[df['is_signal'] == 1]
    df_bkg = df.loc[df['is_signal'] == 0]
    if ax == None:
        _, ax = plt.subplots()
    ax.hist(df_sig[feature_name], range=limits, bins=bins, color='red', label='Signal', alpha=0.5)
    ax.hist(df_bkg[feature_name], range=limits, bins=bins, color='blue', label='Background', alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
