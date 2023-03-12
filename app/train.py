import sys
sys.path.append('./')
import argparse
import pandas
import keras_tuner
from keras.callbacks import EarlyStopping
from hbbclass.dnn.model import HyperDNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import hbbclass.dnn.config as dnn_cfg
import hbbclass.logistic.config as log_cfg


FEATURES = ['Hcand_C2', 'Hcand_D2', 'Hcand_tau21', 'Hcand_tau32']
PRED_LABEL = 'is_signal'


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Parse training arguments.')
    parser.add_argument('model_name', type=str, choices=['dnn', 'logistic'], help='Name of the model to be trained.')
    args = parser.parse_args()
    # Load train and test data
    df_train = pandas.read_csv('./data/processed/train.csv', index_col=0)
    df_test  = pandas.read_csv('./data/processed/test.csv', index_col=0)
    # Extract features and prediction labels
    X_train = df_train[FEATURES]
    y_train = df_train[PRED_LABEL]
    X_test = df_test[FEATURES]
    y_test = df_test[PRED_LABEL]
    # Optimize and train deep neural network model
    if args.model_name == 'dnn':
        # Split train dataset into train and validation samples
        df_train = df_train.sample(frac=0.8, replace=False, random_state=dnn_cfg.SEED)
        df_val   = df_train.drop(df_train.index)
        X_train = df_train[FEATURES]
        y_train = df_train[PRED_LABEL]
        X_val = df_val[FEATURES]
        y_val = df_val[PRED_LABEL]
        # Define early stopping rule
        es = EarlyStopping(
            monitor='val_loss',
            patience=dnn_cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        # Initialize the hyperparameter tuner
        tuner = keras_tuner.Hyperband(
            HyperDNN(),
            objective='val_accuracy',
            max_epochs=dnn_cfg.EPOCHS,
            factor=3,
            seed=dnn_cfg.SEED,
            directory=dnn_cfg.OUTPUT_PATH,
            project_name=dnn_cfg.TUNING_NAME
        )
        # Tune the model
        tuner.search(
            x=X_train, y=y_train,
            validation_data=(X_val, y_val),
            callbacks=[es],
            epochs=dnn_cfg.EPOCHS
        )
        # Print best hyperparameters
        best_hps = tuner.get_best_hyperparameters()[0]
        print('\nBest hyperparameters:')
        print(best_hps.values)
        # Train model with best hyperparameters
        model = HyperDNN().build(best_hps)
        model.fit(X_train, y_train, epochs=dnn_cfg.EPOCHS, validation_data=(X_test, y_test), callbacks=[es])
        # Evaluate model on test sample
        print('\nPerformance of deep neural network model on test sample:')
        model.evaluate(X_test, y_test)
        # Save tuned model
        model.save(f'{dnn_cfg.OUTPUT_PATH}/{dnn_cfg.MODEL_NAME}.model')
    # Optimize and train logistic regression model
    elif args.model_name == 'logistic':
        model = LogisticRegression()
        model = GridSearchCV(model, param_grid=log_cfg.hyper_pars, cv=5, verbose=3, n_jobs=-1)
        model.fit(X_train, y_train)
        print('\nBest hyperparameters:')
        print(model.best_params_)
        print('\nPerformance of logistic regression on test sample:')
        y_pred = model.predict(X_test)
        print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
