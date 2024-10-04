# Import necessary libraries
import os
import sys
# sys.path.append('d:\\data sceince with python\\model_helpers')
sys.path.append(os.path.abspath('') + os.path.sep + 'model_helpers')
from tensorflow import keras
from variable_config import LOOK_BACK as look_back, EPOCHS, file_name
from data_wrangler import data_wrangler, split_into_datasets
from model_builder import hb_tuner

def train_model():

    # Call data_wrangler to create features and label
    X, y, data, scaler = data_wrangler(file_name, look_back)

    # Hold out validation data
    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, data, look_back)
    X_train, X_val, X_test, y_train, y_val, y_test = split_into_datasets(X=X, y=y, look_back=look_back, get_val_set=True)
    
    # Create determinism and model reproducibility
    # tf.keras.utils.set_random_seed(1)
    # tf.config.experimental.enable_op_determinism()

    # Early stoping callbacks for best epoch
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, mode='min')

    check_point = keras.callbacks.ModelCheckpoint(
        # 'hyper_model/check_point/{epoch:03d}-{val_loss:.4f}.keras',
        # 'hyper_model/check_point/epoch_{epoch:03d}.keras',
        'hyper_model/check_point/best_check_point.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'
    )

    # Scheduler function for LearningRateScheduler
    # def scheduler(epoch, lr):
    #     return lr
    # lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    # callbacks = [early_stop, lr_scheduler]
    # hb_tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop, check_point])
    hb_tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop, check_point])

    # print('Search Space Summary:\n', hb_tuner.search_space_summary())
    print('Result Summary:\n', hb_tuner.results_summary())

    # Returns the best hyperparameters, as determined by the objective.
    # These hyperparameters can be used to reinstantiate the (untrained) best model found during the search process.
    best_hps = hb_tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

    # Get the top 1 model.
    # best_model = tuner.get_best_models(num_models=1)[0]
    # Build the model with the optimal hyperparameters and train it on the data for user defined epochs
    model = hb_tuner.hypermodel.build(best_hps)
    # For best performance, it is recommended to retrain your Model on the full dataset using the best hyperparameters found during search
    # history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop])
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stop])

    # val_loss_per_epoch = model.history['val_loss']
    # best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    # print('Best epoch: %d' % (best_epoch))

    # After finding the best epochs, lets re-instantiate the hypermodel and  retrain it.
    # best_model = hb_tuner.hypermodel.build(best_hps)
    # best_model.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_val, y_val))
    # best_model.summary()
    print('Model History:\n', history.history)
    model.summary()

    # Saves the entire model in new high-level .keras format
    model.save('hyper_model/best_model/best_model.keras')
    # best_model.save('hyper_model/best_model/best_model.h5')
    # Saves the entire model as a SavedModel. It places the contents of model in a directory.
    # best_model.export('hyper_model/best_model/best_model')

    # return hb_tuner, best_hps, best_epoch, best_model
    return hb_tuner, best_hps, model

train_model()