import sys
sys.path.append('./')
import pandas
from hbbclass.preprocessing import root_to_pandas, train_test_split


FEATURES = [
    'Hcand_C2',
    'Hcand_D2',
    'Hcand_tau21',
    'Hcand_tau32',
    'Hcand_m',
    'w'
]
N_SAMPLES = 70000


def main():
    # File names of raw data
    file_names = {
        'higgs': './data/raw/Higgs_srl.root',
        'ttbar': './data/raw/ttbar_PowPy8_srl.root'
    }
    # Preprocess raw data
    dataframes = {}
    for process_name, file_name in file_names.items():
        # Convert ROOT file to pandas DataFrame
        df = root_to_pandas(file_name, FEATURES)
        # Filter out events with negative weights
        df = df[df['w'] > 0]
        # Sample dataframe according to event weights
        df = df.sample(n=N_SAMPLES, replace=True, weights=df.loc[:, 'w'])
        # Drop weight column
        df = df.drop('w', axis=1)
        # Add prediction label
        is_signal = 1 if process_name == 'higgs' else 0
        df.insert(0, 'is_signal', [is_signal]*len(df.index), True)
        # Store dataframe
        dataframes[process_name] = df
    # Concatenate preprocessed dataframes
    df = pandas.concat(dataframes.values())
    # Shuffle dataframe
    df = df.sample(frac=1)
    # Reset index
    df.reset_index(drop=True, inplace=True)
    # Split into train and test datasets
    df_train, df_test = train_test_split(df, frac_train=0.7)
    # Save preprocessed samples as csv
    df_train.to_csv('./data/processed/train.csv')
    df_test.to_csv('./data/processed/test.csv')


if __name__ == '__main__':
    main()
