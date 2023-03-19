# -*- coding: utf-8 -*-
'''
@Time    : 2019/11/21 23:26
@Author  : Zekun Cai
@File    : preddiffusion_DTM.py
@Software: PyCharm
'''
import sys
import shutil
import os
import time
import numpy as np
from datetime import datetime
import scipy.sparse as ss
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, concatenate
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
from load_data_DTM import *
from metric import *

###########################Reproducible#############################
import random

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.set_random_seed(100)


###################################################################
def my_mse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    return mse_weight * mse


def my_mape(y_true, y_pred):
    mape = 100 * K.mean(K.abs((y_true - y_pred) / K.clip((K.abs(y_pred) + K.abs(y_true)) * 0.5, EPSILON, None)))
    return mape_weight * mape


def mse_mape(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    mape = 100 * K.mean(K.abs((y_true - y_pred) / K.clip((K.abs(y_pred) + K.abs(y_true)) * 0.5, EPSILON, None)))
    loss = mse_weight * mse + mape_weight * mape
    return loss

class iLayer(Layer):
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape

class Attention(Layer):
    def __init__(self, bias=True, **kwargs):
        self.supports_masking = True
        self.name = 'Attention'
        self.bias = bias
        self.step_dim = 0
        self.features_dim = 0
        self.Height = 0
        self.Width = 0
        self.Filter = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5

        self.W = self.add_weight(shape=input_shape[2:],
                                 initializer='glorot_normal',
                                 name='{}_W'.format(self.name))
        self.step_dim = input_shape[1]
        self.features_dim = input_shape[2] * input_shape[3] * input_shape[4]
        self.Height, self.Width, self.Filter = input_shape[2], input_shape[3], input_shape[4]
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))
        else:
            self.b = None
        # self.conv = Conv2D(self.Filter, (1, 1), padding='same', activation='relu')
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, step_dim, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.tile(a[:, :, np.newaxis, np.newaxis, np.newaxis], (1, 1, self.Height, self.Width, self.Filter))

        # x = K.reshape(x, (-1, self.Height, self.Width, self.Filter))
        # x = self.conv(x)
        # x = K.reshape(x, (-1, self.step_dim, self.Height, self.Width, self.Filter))
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.Height, self.Width, self.Filter

class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',  # Gaussian distribution
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        # assert len(features_shape) == 2
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(GraphConvolution, self).build(input_shapes)

    def call(self, inputs, mask=None):
        features = inputs[0]
        links = inputs[1]

        result = K.batch_dot(links, features, axes=[2, 1])
        output = K.dot(result, self.kernel)
        # output = result

        if self.bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def sequence_GCN(input_seq, adj_seq, unit, act='relu', **kwargs):
    GCN = GraphConvolution(unit, activation=act, **kwargs)
    embed = []
    for n in range(input_seq.shape[2]):
        frame = Lambda(lambda x: x[:, :, n, :])(input_seq)
        adj = Lambda(lambda x: x[:, n, :, :])(adj_seq)
        embed.append(GCN([frame, adj]))
    output = Lambda(lambda x: tf.stack(x, axis=2))(embed)
    return output


def getModel():
    x_input = Input(batch_shape=(BATCHSIZE, HEIGHT * WIDTH, TIMESTEP, HEIGHT, WIDTH, CHANNEL))
    X_meta = Input(batch_shape=(BATCHSIZE, day_fea))
    X_adj = Input(batch_shape=(BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH))
    X_seman = Input(batch_shape=(BATCHSIZE, HEIGHT * WIDTH, TIMESTEP, kernel_window, kernel_window, CHANNEL))

    order = []
    for i in range(CHANNEL):
        order_fea = Lambda(lambda x: x[:, :, :, :, :, i])(X_seman)
        order_flatten = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, TIMESTEP,
                                                       kernel_window * kernel_window)))(order_fea)
        order_gcn = sequence_GCN(order_flatten, X_adj, HEIGHT * WIDTH)
        order.append(order_gcn)
    X_global = Lambda(lambda x: tf.stack(x, axis=-1))(order)
    X_global = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, TIMESTEP, HEIGHT, WIDTH, CHANNEL)))(X_global)

    dens1 = Dense(units=10, activation='relu')(X_meta)
    dens2 = Dense(units=TIMESTEP * HEIGHT * WIDTH * 1, activation='relu')(dens1)
    hmeta = Reshape((TIMESTEP, HEIGHT, WIDTH, 1))(dens2)
    hmeta = Lambda(lambda x: K.concatenate([x[:, np.newaxis, :, :, :, :]] * HEIGHT * WIDTH, axis=1))(hmeta)

    x = concatenate([x_input, X_global, hmeta], axis=-1)

    x = Lambda(lambda x: K.reshape(x, (-1, TIMESTEP, HEIGHT, WIDTH, CHANNEL * 2 + 1)))(x)
    x = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True, activation='tanh')(x)

    outs = []
    for i in range(CHANNEL):
        dens = Dense(units=10, activation='relu')(X_meta)
        dens = Dense(units= HEIGHT * WIDTH * 1, activation='relu')(dens)
        hmeta = Reshape((HEIGHT * WIDTH, 1))(dens)
        hmeta = Dense(units=10, activation='relu')(hmeta)
        hmeta = Dense(units=HEIGHT * WIDTH * 1, activation='relu')(hmeta)
        hmeta = Lambda(lambda x: K.reshape(x, (-1, HEIGHT, WIDTH, 1)))(hmeta)

        out = Attention()(x)
        out = concatenate([out, hmeta], axis=-1)
        out = Conv2D(1, (1, 1), padding='same', activation='relu')(out)
        out = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, HEIGHT * WIDTH)))(out)
        out = iLayer()(out)
        out = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, HEIGHT, WIDTH, 1)))(out)
        outs.append(out)
    model = Model(inputs=[x_input, X_meta, X_adj, X_seman], outputs=outs)
    return model


def testModel(name, testData, dayinfo):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = getModel()
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(PATH + '/' + name + '.h5')
    model.summary()

    test_gene = test_generator_window_gcn(testData, dayinfo, batch=BATCHSIZE, return_gcn=True, adj_type='hisflowL1')
    test_step = (testData.shape[0] - TIMESTEP) // BATCHSIZE
    testY = get_true_window_gcn(testData)

    pred = model.predict_generator(test_gene, steps=test_step, verbose=1)
    pred = np.concatenate(pred, axis=-1)
    pred = pred.reshape((-1, HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    print('pred shape: {}'.format(pred.shape))
    pred_sparse = ss.csr_matrix(pred.reshape(pred.shape[0], -1))
    re_pred_sparse, re_testY = pred_sparse * MAX_DIFFUSION, testY * MAX_DIFFUSION
    mse_score = mse_func(re_testY, re_pred_sparse)
    rmse_score = rmse_func(re_testY, re_pred_sparse)
    mae_score = mae_func(re_testY, re_pred_sparse)
    mape_score = mape_func(re_testY, re_pred_sparse)

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Rescaled MSE on testData, {}\n".format(mse_score))
    f.write("Rescaled RMSE on testData, {}\n".format(rmse_score))
    f.write("Rescaled MAE on testData, {}\n".format(mae_score))
    f.write("Rescaled MAPE on testData, {:.3f}%\n".format(100 * mape_score))
    f.close()

    print('*' * 40)
    print('MSE', mse_score)
    print('RMSE', rmse_score)
    print('MAE', mae_score)
    print('MAPE {:.3f}%'.format(100 * mape_score))
    print('Model Evaluation Ended ...', time.ctime())

    predictionDiffu = re_pred_sparse
    groundtruthDiffu = re_testY
    ss.save_npz(PATH + '/' + MODELNAME + '_prediction.npz', predictionDiffu)
    ss.save_npz(PATH + '/' + MODELNAME + '_groundtruth.npz', groundtruthDiffu)


def trainModel(name, trainData, dayinfo):
    print('Model Training Started ...', time.ctime())

    train_num = int(trainData.shape[0] * (1 - SPLIT))
    print('train num: {}, val num: {}'.format(train_num, trainData.shape[0] - train_num))

    train_gene = data_generator_window_gcn(trainData[:train_num], dayinfo[:train_num], BATCHSIZE, return_gcn=True,
                                           return_sep=True, adj_type='hisflowL1')
    val_gene = data_generator_window_gcn(trainData[train_num:], dayinfo[train_num:], BATCHSIZE,
                                         return_gcn=True, return_sep=True, adj_type='hisflowL1')
    train_step = (train_num - TIMESTEP) // BATCHSIZE
    val_step = (trainData.shape[0] - train_num - TIMESTEP) // BATCHSIZE

    ###single
    model = getModel()
    model.compile(loss=mse_mape, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit_generator(train_gene, steps_per_epoch=train_step, epochs=EPOCH,
                        validation_data=val_gene, validation_steps=val_step,
                        callbacks=[csv_logger, checkpointer, LR, early_stopping])
    pred = model.predict_generator(test_generator_window_gcn(trainData[train_num:], dayinfo[train_num:],
                                                             BATCHSIZE, return_gcn=True, adj_type='hisflowL1'),
                                   steps=val_step)
    pred = np.concatenate(pred, axis=-1)
    pred = pred.reshape((-1, HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    print('pred shape: {}'.format(pred.shape))
    pred_sparse = ss.csr_matrix(pred.reshape(pred.shape[0], -1))
    valY = get_true_window_gcn(trainData[train_num:])

    re_pred_sparse, re_valY = pred_sparse * MAX_DIFFUSION, valY * MAX_DIFFUSION
    mse_score = mse_func(re_valY, re_pred_sparse)
    rmse_score = rmse_func(re_valY, re_pred_sparse)
    mae_score = mae_func(re_valY, re_pred_sparse)
    mape_score = mape_func(re_valY, re_pred_sparse)

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Rescaled MSE on valData, {}\n".format(mse_score))
    f.write("Rescaled RMSE on valData, {}\n".format(rmse_score))
    f.write("Rescaled MAE on valData, {}\n".format(mae_score))
    f.write("Rescaled MAPE on valData, {:.3f}%\n".format(100 * mape_score))
    f.close()

    print('*' * 40)
    print('MSE', mse_score)
    print('RMSE', rmse_score)
    print('MAE', mae_score)
    print('MAPE {:.3f}%'.format(100 * mape_score))
    print('Model Train Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'temporal_att_gcn_hisflowL1'
KEYWORD = 'preddiffusion_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M%S")
PATH = '../' + KEYWORD


################# Parameter Setting #######################


def main():
    param = sys.argv
    if len(param) == 2:
        GPU = param[-1]
    else:
        GPU = '2'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.Session(graph=tf.get_default_graph(), config=config))

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('load_data_DTM.py', PATH)
    shutil.copy2('metric.py', PATH)

    diffusion_data = ss.load_npz(diffusionFile)
    diffusion_data = diffusion_data / MAX_DIFFUSION
    dayinfo = np.genfromtxt(dayinfoFile, delimiter=',', skip_header=1)
    print('data.shape, dayinfo.shape', diffusion_data.shape, dayinfo.shape)
    train_Num = int(diffusion_data.shape[0] * trainRatio)

    print(KEYWORD, 'training started', time.ctime())
    trainvalidateData = diffusion_data[:train_Num]
    trainvalidateDay = dayinfo[:train_Num, ]
    print('trainvalidateData.shape', trainvalidateData.shape)
    trainModel(MODELNAME, trainvalidateData, trainvalidateDay)

    print(KEYWORD, 'testing started', time.ctime())
    testData = diffusion_data[train_Num:]
    testDay = dayinfo[train_Num:]
    print('testData.shape', testData.shape)
    testModel(MODELNAME, testData, testDay)


if __name__ == '__main__':
    main()
