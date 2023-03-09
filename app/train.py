import sys
sys.path.append('./')
import pandas
import keras_tuner
from keras.callbacks import EarlyStopping
from hbbclass.dnn.model import HyperDNN
import hbbclass.dnn.config as cfg


def main():
    # Load training data
    df = pandas.read_csv('./data/processed/train.csv', index_col=0)
    # Get train, validation and test datasets
    df_train = df.sample(frac=0.8, replace=False, random_state=cfg.SEED)
    df_val   = df.drop(df_train.index)
    df_test  = pandas.read_csv('./data/processed/test.csv', index_col=0)
    # Extract features and prediction labels
    features = ['Hcand_C2', 'Hcand_D2', 'Hcand_tau21', 'Hcand_tau32']
    pred_label = 'is_signal'
    X_train = df_train[features]
    y_train = df_train[pred_label]
    X_val = df_val[features]
    y_val = df_val[pred_label]
    X_test = df_test[features]
    y_test = df_test[pred_label]
    # Define early stopping rule
    es = EarlyStopping(
        monitor='val_loss',
        patience=cfg.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    # Initialize the hyperparameter tuner
    tuner = keras_tuner.Hyperband(
		HyperDNN(),
		objective='val_accuracy',
		max_epochs=cfg.EPOCHS,
		factor=3,
		seed=cfg.SEED,
		directory=cfg.OUTPUT_PATH,
		project_name=cfg.TUNING_NAME
    )
    # Tune the model
    tuner.search(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        callbacks=[es],
        epochs=cfg.EPOCHS
    )
    # Print best hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print('\nBest hyperparameters:')
    print(best_hps.values)
    # Train model with best hyperparameters
    model = HyperDNN().build(best_hps)
    model.fit(X_train, y_train, epochs=cfg.EPOCHS, validation_data=(X_test, y_test), callbacks=[es])
    # Evaluate model on test sample
    print('\nModel performance on test sample:')
    model.evaluate(X_test, y_test)
    # Save tuned model
    model.save(f'{cfg.OUTPUT_PATH}/{cfg.MODEL_NAME}.model')


if __name__ == '__main__':
    main()
