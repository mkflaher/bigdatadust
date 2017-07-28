import numpy as np
import tensorflow as tf
import csv

seed = 56
tf.set_random_seed(seed)
def init(shape):
    weights = tf.random_normal(shape,stddev=0.1)
    return tf.Variable(weights)
def forwardprop(X,w1,w2,w3,w4,w5,w6,w7,w8,w9): #calculates the neural network
    #input already contains a bias. need to give bias to the outputs of each layer
    h1 = tf.nn.tanh(tf.matmul(X,w1))
    h2 = tf.nn.tanh(tf.matmul(h1,w2))
    h3 = tf.nn.tanh(tf.matmul(h2,w3))
    h4 = tf.nn.tanh(tf.matmul(h3,w4))
    h5 = tf.nn.tanh(tf.matmul(h4,w5))
    h6 = tf.nn.tanh(tf.matmul(h5,w6))
    h7 = tf.nn.tanh(tf.matmul(h6,w7))
    h8 = tf.nn.tanh(tf.matmul(h7,w8))
    yhat = tf.nn.sigmoid(tf.matmul(h8,w9))
    return yhat

def main(): #this will do all the work
    #This is just a simple feedfforward neural network that I'll write before a RNN.
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

    test_x = np.zeros(pclen)
    test_y = np.zeros(2)
    with open('testpc.csv') as testpc:
        csvr = csv.reader(testpc, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
        print(pclen)
        test_x = np.zeros(pclen)
        for row in csvr:
            vec = np.array([float(x) for x in row[1:]])
            test_x = np.vstack((test_x, np.append(vec,1)))
            #train_y = np.append(train_y,float(row[0])) #1 means there was a dust event, 0 means none
            if float(row[0])==1: #this if-else statement is for logits
                test_y = np.vstack((test_y,[0,1]))
            else:
                test_y = np.vstack((test_y,[1,0]))
    #train_y = train_y.reshape(-1,1)
    #with open('ndustpc.csv') as ndustpc:
    #    csvr = csv.reader(ndustpc, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
    #    for row in csvr:
    #        vec = np.array([float(x) for x in row])
    #        train_x = np.vstack((train_x, np.append(vec,1)))
    #        train_y = np.append(train_y,0).reshape((-1,1)) #0 means no dust event


    train_x = train_x[1:]
    train_y = train_y[1:]
    test_x = test_x[1:] #chop of first entry since it's all zeros
    test_y = test_y[1:] #uncomment for logits
 
    #set up the network
    nNodes = 25
    #We need three matrices for weights: inputs to layer 1, layer 1 to layer 2, layer 2 to output
    w1 = init((pclen,nNodes))
    w2 = init((nNodes,nNodes))
    w3 = init((nNodes,nNodes))
    w4 = init((nNodes,nNodes))
    w5 = init((nNodes,nNodes))
    w6 = init((nNodes,nNodes))
    w7 = init((nNodes,nNodes))
    w8 = init((nNodes,nNodes))
    w9 = init((nNodes,2))
    print(pclen)
    X = tf.placeholder("float",shape=[None,pclen]) #Create a placeholder for the principal components as inputs

    y = tf.placeholder("float",shape=[None,2]) #Outputlayer is a binary classification, so we use a node that either does or doesn't fire

    yhat = forwardprop(X,w1,w2,w3,w4,w5,w6,w7,w8,w9)
    predict = tf.argmax(yhat,axis=1)
    logit_weights = tf.constant([1.0,7.9064])
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=yhat,targets=y,pos_weight=logit_weights)
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    sess = tf.Session() #create and run session
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    fflog = open('fflog.txt','w')

    falsePos = 0
    falseNeg = 0
    total = 1
    for epoch in range(1000):
        #train with examples 100 times
        for i in range(len(train_x)):
            sess.run(updates, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]})
            if(epoch==999):
                #fflog.write('Dust Confidence: ', sess.run(yhat,feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]}), 'Actual: ', train_y[i:i+1],'\r\n')
                fflog.write('Dust Confidence: %s Actual: %s ' % (sess.run(yhat,feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]}),train_y[i:i+1]))
                if np.argmax(sess.run(yhat,feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]}))==np.argmax(train_y[i:i+1]):
                    fflog.write('Correct! ')
                else:
                    fflog.write('WRONG! ')
                fflog.write('Loss: %s' % sess.run(loss, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]}))
                fflog.write('\r\n')
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict,feed_dict={X:train_x,y:train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict,feed_dict={X:test_x,y:test_y}))
        print("Epoch %d  Training Accuracy %.9f  Test Accuracy %.9f%%" % (epoch + 1, 100*train_accuracy, 100*test_accuracy))


    summary_writer = tf.summary.FileWriter('/home/sean/proj/dust/tblog', sess.graph)
    fflog.close()

if __name__ == '__main__':
    main()
