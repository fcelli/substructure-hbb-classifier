import sys
sys.path.append('./')
import pandas
from hbbclass.preprocessing import root_to_pandas, train_test_sample
# Random seed for reproducible results
seed = 42


def main():
    # Define variable names
    columns = [
        'Hcand_m',
        'Hcand_C2',
        'Hcand_D2',
        'Hcand_tau21',
        'Hcand_tau32',
        'w'
    ]
    # Load signal sample
    df_sig = root_to_pandas('./data/raw/Higgs_srl.root', columns)
    # Load backgound sample
    df_bkg = root_to_pandas('./data/raw/ttbar_PowPy8_srl.root', columns)
    # Drop events with negative values
    df_sig = df_sig[df_sig['w'] >= 0]
    df_bkg = df_bkg[df_bkg['w'] >= 0]
    # Drop events with NaN feature values
    df_sig.dropna(axis=0, inplace=True)
    df_bkg.dropna(axis=0, inplace=True)
    # Add prediction label
    if 'is_signal' not in df_sig:
        df_sig.insert(0, 'is_signal', [1]*len(df_sig.index), True)
    if 'is_signal' not in df_bkg:
        df_bkg.insert(0, 'is_signal', [0]*len(df_bkg.index), True)
    # split samples into train and test datasets
    n_samples = 70000
    frac_train = 0.7
    df_sig_train, df_sig_test = train_test_sample(df_sig, n_samples, frac_train, 'w', random_state=seed)
    df_bkg_train, df_bkg_test = train_test_sample(df_bkg, n_samples, frac_train, 'w', random_state=seed)
    # Concatenate dataframes
    df_train = pandas.concat([df_sig_train, df_bkg_train])
    df_test = pandas.concat([df_sig_test, df_bkg_test])
    # Shuffle dataframes
    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)
    # Reset indexes
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    # Some sanity checks
    assert df_train.shape[0] == int(2*n_samples*frac_train)
    assert df_test.shape[0] == int(2*n_samples*(1-frac_train))
    # Save preprocessed samples to csv files
    df_train.to_csv('./data/processed/train.csv')
    df_test.to_csv('./data/processed/test.csv')


if __name__ == '__main__':
    main()
