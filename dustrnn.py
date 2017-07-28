#A modified LSTM cell that does a binary prediction by taking the cell state and putting it through a single weighted sum tanh node.

import tesnorflow as tf
import csv
import numpy as np

def initWeights(shape):
    weights = tf.random_normal(shape,stddev=0.1)
    return tf.Variable(weights)
    #sets the weights for components in the RNN

def sigmoid_cell(X,nInputs,nLayers,nNodes,nOutputs): #Some elements of the RNN will be feedforward network cells, this gives us the infrastructure to quickly and dynamically create them.
    weights = []
    for i in range(nLayers+2):
        if i==0:
            weights.append(initWeights((nInputs,nNodes)))
        elif i==nLayers+1:
            weights.append(initWeights((nNodes,nOutputs)))
        else:
            weights.append(initWeights((nNodes,nNodes)))
    yhat = X
    for weight in weights:
        yhat = tf.nn.sigmoid(tf.matmul(yhat,weight))
    return yhat

def tanh_cell(X,nInputs,nLayers,nNodes,nOutputs): #Some elements of the RNN will be feedforward network cells, this gives us the infrastructure to quickly and dynamically create them.
    #nInputs is included to avoid complications of relying on tensorflow's lazy evaluation to give the input size of a tensor.
    weights = []
    for i in range(nLayers+2):
        if i==0:
            weights.append(initWeights((nInputs,nNodes)))
        elif i==nLayers+1:
            weights.append(initWeights((nNodes,nOutputs)))
        else:
            weights.append(initWeights((nNodes,nNodes)))
    yhat = X
    for weight in weights:
        yhat = tf.nn.tanh(tf.matmul(yhat,weight))
    return yhat

def lstm_cell(hprev,Cprev,Xin,nInputs,nLayers,nNodes,nOutputs):
    inlayer = tf.concat([hprev,X],1) #concatenate horizontally
    #first step: forget gate: f_t = sigmoid(W_i*[h_t-1, x_t])
    f_t = sigmoid_cell(inlayer,nInputs,nLayers,nNodes,nOutputs)
    #input layer gate
    i_t = sigmoid_cell(inlayer,nInputs,nLayers,nNodes,nOutputs)
    cand_t = tanh_cell(inlayer,nInputs,nLayers,nNodes,nOutputs)

    #Now let's update the cell state
    cell_state = Cprev * f_t + i_t * cand_t

    #make a new h_t
    o_t = sigmoid_cell(inlayer,nInputs,nLayers,nNodes,nOutputs)
    #this is the predictor we will use to train and test
    h_t = tf.tanh(cell_state) * o_t

    return h_t, cell_state #these get saved to go into the next LSTM cell


def main():
    #Retrieve the training data
    #numpy will preprocess the data into an input format for training
    #preprocessing
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

    
 

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    for epoch in range(100): #how many times I want to run through the training data:
        for i in range(len(train_x)):
            sess.run(updates,feed_dict={xin:train_x[i:i+1],y:train_y[i:i+1])

