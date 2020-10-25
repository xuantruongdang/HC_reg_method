from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

class Model():
    def reg_resnet50(self):
        """
        input: 224 x 224 x 3
        optimizer: Adam
        learning rate: 1e-3
        loss: Mean square error
        metric: Mean absolute error

        """ 
        res = ResNet50(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet', pooling='avg')
        model = Sequential()
        model.add(res)
        # model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='linear'))
        optimizer = Adam(learning_rate=1e-3, amsgrad=True)
        model.compile(optimizer = optimizer, loss = 'MSE', metrics = ['mean_absolute_error'])
        return model, optimizer