import keras
import keras_tuner
import hbbclass.dnn.config as cfg


class HyperDNN(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=cfg.INPUT_SHAPE))
        n_layers = hp.Int('n_layers', cfg.N_LAYERS_LOW, cfg.N_LAYERS_HIGH)
        for i in range(n_layers):
            n_units = hp.Int(f'layer{i+1}_units', cfg.N_UNITS_LOW, cfg.N_UNITS_HIGH, cfg.N_UNITS_STEP)
            model.add(keras.layers.Dense(n_units, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        learning_rate = hp.Choice('learning_rate', values=cfg.LEARNING_RATES)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', cfg.BATCH_SIZE),
            **kwargs,
        )
