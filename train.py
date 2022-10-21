import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import newaxis

from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv

from pre_processing import PreProcessing

# Predict SLB stock price from news and stock data
# DP-LSTM: Differential Privacy-inspired LSTM for Stock Prediction Using Financial News

class LSTMWithSentiment(object):
    
    def __init__(self):
        
        self.split = 1
        self.sequence_length=10
        self.normalise= True
        self.batch_size=32
        self.input_dim=3
        self.input_timesteps=self.sequence_length-1
        self.neurons=50
        self.epochs=1000
        self.prediction_len=1
        self.dense_output=1
        self.drop_out=0.05

    def SplitTrainTest(self, pp):
        
        recordDict = {}
        dataDict = {}

        DataModel = pp.final_data.copy()
        i_split = int(len(DataModel) * self.split)

        cols = ['price','date','wsj']

        if self.split == 1:
            # train with all the data
            data_train_1 = DataModel.get(cols).values[:]
            # use last sequence to generate input for future prediction
            data_test = DataModel.get(cols).values[:][-self.sequence_length:]
            dataDict["data_test"] = data_test
        else:
            data_train_1 = DataModel.get(cols).values[:i_split]
            data_test = DataModel.get(cols).values[i_split:]

        X_train, y_train = self.SplitSample(data_train_1)
        record_min, record_max, X_test, y_test = self.SplitSample(data_test, getRecord=True)
        _, y_test_ori = self.SplitSample(data_test, normalise=False)

        recordDict["min"] = record_min
        recordDict["max"] = record_max
        dataDict["x_train"] = X_train
        dataDict["y_train"] = y_train
        dataDict["x_test"] = X_test
        dataDict["y_test"] = y_test
        dataDict["y_test_ori"] = y_test_ori

        return dataDict, recordDict

    def SplitSample(self, data, normalise=True, getRecord=False):
        
        window = self.GenerateWindow(data, self.sequence_length)
        if not normalise:
            X = window[:,:-self.prediction_len]
            y = window[:, -self.prediction_len:, [0]]
            if getRecord:
                record_min, record_max, norm_data = self.ComputeNormalisedDataAndRecord(window)
        else:
            record_min, record_max, norm_data = self.ComputeNormalisedDataAndRecord(window)
            X = norm_data[:,:-self.prediction_len]
            y = norm_data[:, -self.prediction_len:, [0]]
            y = np.reshape(y, (len(norm_data),self.prediction_len))
        
        if getRecord:
            return record_min, record_max, X, y
        return X, y
        
    def GenerateWindow(self, data, seqlen):

        # rolling window of size "seqlen"
        len_data = len(data)
        window_data = []

        if len_data == seqlen:
            n_samples = 1
            window_data = data.reshape(n_samples,data.shape[0], data.shape[1])
        else:
            for i in range(len_data - seqlen):
                window_data.append(data[i:i+seqlen])
            window_data = np.array(window_data).astype(float)

        return window_data

    def ComputeNormalisedDataAndRecord(self, window_data):

        win_num = window_data.shape[0]
        col_num = window_data.shape[2]

        normalised_data = []
        record_min = []
        record_max = []

        for win_i in range(0,win_num):
            normalised_window = []
            for col_i in range(0,1):#col_num):
                temp_col = window_data[win_i,:,col_i]
                temp_min = min(temp_col)
                #temp_min=0
                if col_i == 0:
                    record_min.append(temp_min)#record min
                temp_col = temp_col - temp_min
                temp_max = max(temp_col)
                #temp_max=1
                if col_i == 0:
                    record_max.append(temp_max)#record max
                temp_col = temp_col/temp_max
                normalised_window.append(temp_col)
            for col_i in range(1,col_num):
                temp_col = window_data[win_i,:,col_i]
                normalised_window.append(temp_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        normalised_data = np.array(normalised_data)
        return record_min, record_max, normalised_data

    def TrainLSTM(self, x_train, y_train):

        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(self.input_timesteps, self.input_dim), return_sequences = True))
        model.add(Dropout(self.drop_out))
        model.add(LSTM(self.neurons,return_sequences = True))
        model.add(LSTM(self.neurons,return_sequences =False))
        model.add(Dropout(self.drop_out))
        model.add(Dense(self.dense_output, activation='linear'))
        # Compile model
        model.compile(loss='mean_squared_error',
                        optimizer='adam')
        model.fit(x_train,y_train,epochs=self.epochs,batch_size=self.batch_size)

        return model

    def GeneratePrediction(self, model, data, recordDict):
        
        record_min = recordDict["min"]
        record_max = recordDict["max"]
        prediction_seqs = []
        window_size = self.sequence_length
        pre_win_num = int(len(data)/self.prediction_len)

        for i in range(0,pre_win_num):
            curr_frame = data[i*self.prediction_len]
            predicted = []
            for j in range(0,self.prediction_len):
                temp = model.predict(curr_frame[newaxis,:,:])[0]
                predicted.append(temp)
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)

        de_predicted=[]

        m=0
        for i in range(0,pre_win_num):
            for j in range(0,self.prediction_len):
                de_predicted.append(prediction_seqs[i][j][0]*record_max[m]+record_min[m])
                m=m+1

        return de_predicted

    def ComputeMetrics(self, y_test_orig, y_test, de_predicted, len_data):
        
        error = []
        diff=y_test.shape[0]-len_data
        for i in range(y_test_orig.shape[0]-diff):
            error.append(y_test_orig[i,] - de_predicted[i])

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val) 
            absError.append(abs(val))
        
        error_percent=[]
        for i in range(len(error)):
            val=absError[i]/y_test_orig[i,]
            val=abs(val)
            error_percent.append(val)

        mean_error_percent=sum(error_percent) / len(error_percent)
        accuracy=1-mean_error_percent

        MSE=sum(squaredError) / len(squaredError)

        print("MSE",MSE)
        print('accuracy',accuracy)
        print('mean_error_percent',mean_error_percent)

    def PredictFuture(self, model, data):
        
        n_ahead = 10 # predict 2 weeks ahead (business days)
        y_pred = []
        y_test = []
        newData = data.copy()
        for k in range(n_ahead):
            
            record_min, record_max, X_norm, _ = self.SplitSample(newData, getRecord=True, normalise=True)
            curr_frame = X_norm.copy()
            weekendFactor = 1
            if k == 0 or not k % 5:
                weekendFactor = 3
            lastDay = newData[-1,-2]

            pred = model.predict(curr_frame)[0]
            pred_raw = pred*record_max+record_min
            y_pred.append(pred_raw)
            y_test.append(newData[-1,-3])
            
            inputRow = np.append(pred_raw,[lastDay+weekendFactor, 0.0])
            newData = np.insert(newData[1:], [self.sequence_length-1], inputRow, axis=0)

        return y_pred, y_test

def run():
    stockLstm = LSTMWithSentiment()
    forceLoad = True # set this to true if you want to run preprocessing from zero
    pp = PreProcessing(forceLoad)
    
    dataDict, recordDict = stockLstm.SplitTrainTest(pp)

    model = stockLstm.TrainLSTM(dataDict["x_train"], dataDict["y_train"])

    if stockLstm.split == 1:
        # predict future
        y_pred, y_test = stockLstm.PredictFuture(model, dataDict["data_test"])
        y_pred = np.hstack(y_pred)
        print(y_pred)
    else:
        # check the model
        de_predicted = stockLstm.GeneratePrediction(model, dataDict["x_test"], recordDict)
        stockLstm.ComputeMetrics(dataDict["y_test_ori"], dataDict["y_test"], de_predicted, len(dataDict["x_test"]))
        plotModel(de_predicted, dataDict["y_test_ori"])

def plotModel(y_pred, y_test_ori):
    y_test = list(np.reshape(y_test_ori, (y_test_ori.shape[0])))
    plt.plot(y_pred, label = 'predicted')
    plt.plot(y_test, label = 'real')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()