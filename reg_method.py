import datetime
import os
import pandas as pd
import argparse

from dataloader import DataLoader
from models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

"""## Experiment B"""

# resize to 224x224
# normalize by subtracting the mean and dividing by standard deviation
# HC values are normalized by dividing the maximum value of HC
class Training():
    def __init__(self):
        self.train_gen = None
        self.valid_gen = None


    def load_data(self, train_path, valid_path):
        train_set = DataLoader(train_path, mode="train")
        train_gen = train_set.data_gen(32, shuffle=True)
    
        valid_set = DataLoader(valid_path, mode="valid")
        valid_gen = valid_set.data_gen(32, shuffle=True)
        self.train_gen = train_gen
        self.valid_gen = valid_gen


    def time_to_timestr(self):
        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return timestr


    def train(self, model, optimizer):
        early = EarlyStopping(monitor="val_loss",
                              min_delta=1e-4,
                              patience=50,
                              verbose=1)

        anne = ReduceLROnPlateau(monitor="loss",
                                 factor=0.1,
                                 patience=20,
                                 verbose=1,
                                 min_lr=1e-5)

        timestr = self.time_to_timestr()
        log_dir = "./reg_method/logs/fit/reg_method_{}".format(timestr)
        tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)

        os.mkdir("./reg_method/models/reg_method_{}".format(timestr))
        file_path = "./reg_method/models/reg_method_%s/%s_mse={val_loss:.2f}_%s_ep{epoch:02d}.hdf5" % (
            timestr,
            model._name,
            optimizer._name
        )

        checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

        callbacks_list = [early, anne, checkpoint, tensorboard_callback]

        history = model.fit(self.train_gen, 
                            epochs=200,
                            validation_data=self.valid_gen, 
                            callbacks=callbacks_list,
                            workers=10,
                            use_multiprocessing=True)

        his = pd.DataFrame(history.history)
        his.to_csv("./reg_method/models/reg_method_{}/history.csv".format(timestr), index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model, optimizer = Model().reg_resnet50()
    train = Training()
    train.load_data(args.train_path, args.valid_path)
    train.train(model, optimizer)