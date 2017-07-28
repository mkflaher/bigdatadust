#This code borrows mostly from Erik Hallstrom's LSTM tutorial.

import tensorflow as tf
import numpy as np
import csv

def main():
    train_x = np.array([]) #we get this from reading the csv files
    #train_y = np.array([]) #if we're reading the dust file, append 1, 0 for nondust
    train_y = np.zeros(2) #if we're using logits, uncomment this line
    pclen = 0
    with open('trainpc.csv') as trainpc:
        csvr = csv.reader(trainpc, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
        pclen = len(next(csvr)) #how many principal components we're using plus a bias
        print(pclen)
        train_x = np.zeros(pclen)
        for row in csvr:
            vec = np.array([float(x) for x in row[1:]])
            train_x = np.vstack((train_x, np.append(vec,1)))
            #train_y = np.append(train_y,float(row[0])) #1 means there was a dust event, 0 means none
            if float(row[0])==1: #this if-else statement is for logits
                train_y = np.vstack((train_y,[0,1]))
            else:
                train_y = np.vstack((train_y,[1,0]))

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size) #need to figure out what lstm_size does
    state = tf.zeros([batch_size, lstm.state_size]) #initial state of model
    probabilities = []
    loss = 0.0
    for current_batch in batches:
        output, state = lstm(current_batch, state)

    for i i range(
