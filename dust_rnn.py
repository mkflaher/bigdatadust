#Taking feedforward NN code and repurposing it for a simple LSTM cell
import numpy as np
import tensorflow as tf
import csv

seed = 56
tf.set_random_seed(seed)
def init(shape):
    weights = tf.random_normal(shape,stddev=1.0)
    return tf.Variable(weights)

def forwardprop(X,cell,weights_l1,weights_l2):
    #input already contains a bias. need to give bias to the outputs of each layer
    #yhat = X
    #for weight in weights[:-1]:
    #    yhat = tf.nn.tanh(tf.matmul(yhat,weight)) #tanh activation function ensures steeper gradients
    #yhat = tf.nn.sigmoid(tf.matmul(yhat,weights[-1]))
    #return yhat
    forget_l1 = tf.nn.tanh(tf.matmul(X,weights_l1['forget']))
    inp_l1 = tf.nn.tanh(tf.matmul(X,weights_l1['input']))
    cand_l1 = tf.nn.tanh(tf.matmul(X,weights_l1['candidate']))
    out_l1 = tf.nn.tanh(tf.matmul(X,weights_l1['out']))

    forget = tf.nn.sigmoid(tf.matmul(forget_l1,weights_l2['forget']))
    inp = tf.nn.sigmoid(tf.matmul(inp_l1,weights_l2['input']))
    cand = tf.nn.tanh(tf.matmul(cand_l1,weights_l2['candidate']))
    out = tf.nn.sigmoid(tf.matmul(out_l1,weights_l2['out']))


    cell_out = forget * cell + inp * cand #update the cell state
    yhat = tf.tanh(cell_out) * out

    return yhat, cell_out
    

def main(): #this will do all the work
    #This is just a simple feedfforward neural network that I'll write before a RNN.
    #numpy will preprocess the data into an input format for training
    #preprocessing
    train_cell = np.zeros(2) #cell states for training data
    train_hidden = np.zeros(2) #hidden state (comes from prediction)
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
            train_cell = np.vstack((train_cell,[0,0]))
            train_hidden = np.vstack((train_hidden,[0,0]))
            #train_y = np.append(train_y,float(row[0])) #1 means there was a dust event, 0 means none
            if float(row[0])==1: #this if-else statement is for logits
                train_y = np.vstack((train_y,[-1,1]))
            else:
                train_y = np.vstack((train_y,[1,-1]))

    #test_x = np.zeros(pclen)
    #test_y = np.zeros(2)
    #with open('testpc.csv') as testpc:
    #    csvr = csv.reader(testpc, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
    #    print(pclen)
    #    test_x = np.zeros(pclen)
    #    for row in csvr:
    #        vec = np.array([float(x) for x in row[1:]])
    #        test_x = np.vstack((test_x, np.append(vec,1)))
    #        #train_y = np.append(train_y,float(row[0])) #1 means there was a dust event, 0 means none
    #        if float(row[0])==1: #this if-else statement is for logits
    #            test_y = np.vstack((test_y,[0,1]))
    #        else:
    #            test_y = np.vstack((test_y,[1,0]))
    #train_y = train_y.reshape(-1,1)
    #with open('ndustpc.csv') as ndustpc:
    #    csvr = csv.reader(ndustpc, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
    #    for row in csvr:
    #        vec = np.array([float(x) for x in row])
    #        train_x = np.vstack((train_x, np.append(vec,1)))
    #        train_y = np.append(train_y,0).reshape((-1,1)) #0 means no dust event

    #chop off the first zeros
    train_x = train_x[1:]
    train_y = train_y[1:]
    train_cell = train_cell[1:]
    train_hidden = train_hidden[1:]

    train_x = np.hstack((train_hidden,train_x)) #combine the arrays knowing that the first two entries are the LSTM cell hidden state - this makes it easier for the feed dict
    print(len(train_x[0]))
    print(train_x[0])
    #test_x = test_x[1:] #chop of first entry since it's all zeros
    #test_y = test_y[1:] #uncomment for logits
 
    #set up the network
    nNodes = 25
    #We need three matrices for weights: inputs to layer 1, layer 1 to layer 2, layer 2 to output
    nLayers = 5
    nNodes = 25
    #w1 = init((pclen,nNodes))
    #w2 = init((nNodes,nNodes))
    #w3 = init((nNodes,2))
    weights_l1 = {'forget':init((pclen+2,nNodes)),'input':init((pclen+2,nNodes)),'candidate':init((pclen+2,nNodes)),'out':init((pclen+2,nNodes))}
    weights_l2 = {'forget':init((nNodes,2)),'input':init((nNodes,2)),'candidate':init((nNodes,2)),'out':init((nNodes,2))}
    print(pclen)
    X = tf.placeholder("float",shape=[None,pclen+2],name='X') #Create a placeholder for the principal components as inputs
    y = tf.placeholder("float",shape=[None,2],name='y') #Outputlayer is a binary classification, so we use two nodes for weighted cross-entropy
    cell = tf.placeholder("float",shape=[None,2],name='cell') #used for the cell state. Updates the next element in the list.

    yhat,cell_state = forwardprop(X,cell,weights_l1,weights_l2)
    predict = tf.argmax(yhat,axis=1)
    logit_weights = tf.constant([1.0,7.9064])
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=yhat,targets=y,pos_weight=logit_weights)
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    sess = tf.Session() #create and run session
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    rnnlog = open('rnnlog.txt','w')

    for epoch in range(1000):
        #train with examples 100 times
        for i in range(len(train_x)):
            sess.run(updates, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1],cell:train_cell[i:i+1]})
            if not (i==len(train_x)-1): #for all but the last
                train_x[i+1][0:2] = sess.run(yhat, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1],cell:train_cell[i:i+1]})
                train_cell[i+1] = sess.run(cell_state, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1],cell:train_cell[i:i+1]})
                #print(train_x[i+1])
                #print(train_cell[i+1])

            if(epoch==9):
                #rnnlog.write('Dust Confidence: ', sess.run(yhat,feed_dict={X:train_x[i:i+1],y:train_y[i:i+1]}), 'Actual: ', train_y[i:i+1],'\r\n')
                rnnlog.write('Dust Confidence: %s Actual: %s ' % (sess.run(yhat,feed_dict={X:train_x[i:i+1],y:train_y[i:i+1],cell:train_cell[i:i+1]}),train_y[i:i+1]))
                if np.argmax(sess.run(yhat,feed_dict={X:train_x[i:i+1],y:train_y[i:i+1],cell:train_cell[i:i+1]}))==np.argmax(train_y[i:i+1]):
                    rnnlog.write('Correct! ')
                else:
                    rnnlog.write('WRONG! ')
                rnnlog.write('Loss: %s' % sess.run(loss, feed_dict={X:train_x[i:i+1],y:train_y[i:i+1],cell:train_cell[i:i+1]}))
                rnnlog.write('\r\n')
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict,feed_dict={X:train_x,y:train_y,cell:train_cell}))
        #test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict,feed_dict={X:test_x,y:test_y}))
        print("Epoch %d  Training Accuracy %.9f  " % (epoch + 1, 100*train_accuracy))


    summary_writer = tf.summary.FileWriter('tblog', sess.graph)
    rnnlog.close()

if __name__ == '__main__':
    main()
