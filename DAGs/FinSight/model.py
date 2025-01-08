from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential


class Model():
    def get_model(self,type):
        if type == 'hyperparameter':
            return self.hyperparameter_model
        else:
            return self.training_model
    
    def __init__(self):
        self.training_model = Sequential()
        self.hyperparameter_model = Sequential()

    def create_training_layers(self,best_params, input_shape):
        # for _ in range(best_params['num_layers']):
        self.training_model.add(LSTM(units=best_params['units'], return_sequences=True, input_shape=input_shape))
        self.training_model.add(Dropout(rate=best_params['dropout_rate']))
        self.training_model.add(LSTM(units=best_params['units'], return_sequences=True))
        self.training_model.add(Dropout(rate=best_params['dropout_rate']))
        self.training_model.add(LSTM(units=best_params['units'], return_sequences=True))
        self.training_model.add(Dropout(rate=best_params['dropout_rate']))
        self.training_model.add(LSTM(units=best_params['units']))
        self.training_model.add(Dropout(rate=best_params['dropout_rate']))
        self.training_model.add(Dense(units=1))
    
    def hyperparameter_layers(self, units, num_layers, dropout_rate, input_shape):
        for _ in range(num_layers):
            self.hyperparameter_model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
            self.hyperparameter_model.add(Dropout(rate=dropout_rate))
        self.hyperparameter_model.add(LSTM(units=units))
        self.hyperparameter_model.add(Dropout(rate=dropout_rate))
        self.hyperparameter_model.add(Dense(units=1))