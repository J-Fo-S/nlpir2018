"""Module to create model.

Helper functions to create a multi-layer perceptron model and a separable CNN
model. These functions take the model hyper-parameters as input. This will
allow us to create model instances with slightly varying architectures.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy import sparse

import itertools

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

import argparse
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import re

#import build_model
import data_load
#import vectorize_data

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

FLAGS = None

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
        #print(num_classes)
    return units, activation


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as ngram vectors.
    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.
    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.
    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'use_idf': True,
            'ngram_range': FLAGS.NGRAM_RANGE,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word/char tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
            'sublinear_tf': True,
            'norm': 'l2',
            'stop_words': 'english'
    }
    vectorizer = TfidfVectorizer(**kwargs)
    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)
    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)
    
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)
    #print('sample and feature lengths', x_train.shape)  
   
    if FLAGS.word_plus_char == True:
    
        # Create keyword arguments to pass to the 'tf-idf' vectorizer.
        kwargs = {
                'use_idf': True,
                'smooth_idf': True,
                'ngram_range': FLAGS.NGRAM_RANGE,
                'dtype': 'int32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': 'word',  # Split text into word/char tokens.
                'min_df': MIN_DOCUMENT_FREQUENCY,
                'sublinear_tf': True,
                'norm': 'l2',
                'stop_words': 'english',
        }
        
        vectorizer2 = TfidfVectorizer(**kwargs)
        # Learn vocabulary from training texts and vectorize training texts.
        x_train2 = vectorizer2.fit_transform(train_texts)
        # Vectorize validation texts.
        x_val2 = vectorizer2.transform(val_texts)
        
        print(vectorizer2.get_feature_names())

        # Select top 'k' of the vectorized features.
        selector2 = SelectKBest(f_classif, k=min(TOP_K, x_train2.shape[1]))
        selector2.fit(x_train2, train_labels)
        x_train2 = selector2.transform(x_train2)
        x_val2 = selector2.transform(x_val2)
        #print(x_train.shape, x_train2.shape)
        x_train = sparse.hstack((x_train, x_train2)).tocsr()
        x_val = sparse.hstack((x_val, x_val2)).tocsr()

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return x_train, x_val


def train_ngram_model(data,
                      learning_rate=0.001,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=32,
                      dropout_rate=0.4):
    """Trains n-gram model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data
    #print(train_texts[:1],train_labels[:1],val_texts[:1],val_labels[:1])
    print(train_texts[:1])

    # Verify that validation labels are in the same range as training labels.
    num_classes = data_load.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Stem texts by splitting each utterance, not each word. So, split and rejoin.
    stemmer = SnowballStemmer("english")
    train_texts = [" ".join([stemmer.stem(word) for word in sentence.split(" ")]) for sentence in train_texts]   
    val_texts = [" ".join([stemmer.stem(word) for word in sentence.split(" ")]) for sentence in val_texts]    
    #print(train_texts[:1],train_labels[:1],val_texts[:1],val_labels[:1])
    print(train_texts[:1])

    # Vectorize texts.
    x_train, x_val = ngram_vectorize(
        train_texts, train_labels, val_texts)

    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    #class weight for unbalanced classes
    if FLAGS.class_weights == True:
        class_weight = {0: 1.,
                        1: 2.,
                        2: 4.,
                        3: 4.}
    else: 
        class_weight = None

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size,
            class_weight = class_weight)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1])) 
    # Save model.
    model.save('emotion_mlp_model.h5')
    _plot_parameters(history)
    full_multiclass_report(model,
                            x_val,
                            val_labels)
    return history['val_acc'][-1], history['val_loss'][-1]


def _plot_parameters(params, filename='report'):
    """Creates a 3D surface plot of given parameters.
    # Arguments
        params: dict, contains layers, units and accuracy value combinations.
    """
    plt.figure(1)
    epochs = np.linspace(1, len(params['acc']), len(params['acc']))
    plt.subplot(211)
    plt.plot(epochs, params['acc'],'pink', label='train accuracy')
    plt.plot(epochs, params['val_acc'], 'yellow', label='validation accuracy')
    plt.legend(title='accuracy', loc='best', fontsize='medium')
    plt.subplot(212)
    plt.plot(epochs, params['loss'], 'pink', label='train loss')
    plt.plot(epochs, params['val_loss'], 'yellow', label='validation loss')
    plt.legend(title='loss', loc='best', fontsize='medium')
    plt.savefig(filename+'_l_graph')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          filename='report'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename+'confusion')
    plt.show()
    
## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes=['neutral','joy','anger','sadness'],
                           batch_size=128,
                           binary=False,
                           filename='report'):

    # 1. Transform one-hot encoded y_true into their class number
    #if not binary:
       # y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    class_rep = classification_report(y_true,y_pred,digits=5)
    with open(filename+"classification.txt", "w") as f:
        f.write(class_rep)

    print("Classification Report")
    print(class_rep)    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/combined/',
                        help='options= /combined/, /friends/, /emotionPush/')
    parser.add_argument('--NGRAM_RANGE', type=int, default=(2,5),
                        help='input data directory')
    parser.add_argument('--TOKEN_MODE', type=str, default='char_wb',
                        help='options = char, char_wb, word')
    parser.add_argument('--word_plus_char', type=int, default=False,
                        help='options = True or False to combine features')
    parser.add_argument('--class_weights', type=int, default=False,
                        help='options = True or False to balance classes some')
    
    FLAGS, unparsed = parser.parse_known_args()

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 2000

    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    TOKEN_MODE = FLAGS.TOKEN_MODE 

    # Minimum document/corpus frequency below which a token will be discarded.
    MIN_DOCUMENT_FREQUENCY = 2

    # Using the IMDb movie reviews dataset to demonstrate training n-gram model
    data = data_load.load_main(FLAGS.data_dir)
    train_ngram_model(data)
