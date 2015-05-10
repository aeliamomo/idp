"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.
This implementation simplifies the model in the following ways:
 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer
References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""
import os
import sys
import time

import numpy
import matplotlib.cm as cm
import math
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
#from predictionTest import y_pred, predict
import pylab as plt
from numpy import savetxt
from theano.gof.tests import test_optdb
#from logistic_regression import train_set_x



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.1, n_epochs=5,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=9):
    """ Demonstrates lenet on MNIST dataset
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)
    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)
    no_of_layer = 3
    split_train = 9000
    end_train = 13500
    train_set_x_path = 'patch_train_x_42_shuffled_05'
    train_set_y_path = 'patch_train_y_42_shuffled_05'
    
    split_test = 9000
    end_test = 13500
    test_set_x_path = 'patch_train_x_42_shuffled_05'
    test_set_y_path = 'patch_train_y_42_shuffled_05'
    
    patch_size = numpy.loadtxt(train_set_x_path)[:1].shape[1]
    patch_size = int(math.sqrt(patch_size))
    print ("patch size is : ",patch_size)
    
    train_set_x = numpy.loadtxt(train_set_x_path)
    
    print("split",split_train)
    
    
    train_set_x = train_set_x[:split_train]
    #print "read patch :", train_set_x
    print ("train set data : ",train_set_x.shape)
    
    
    train_set_x = theano.shared(numpy.asarray(train_set_x,
                                               dtype=theano.config.floatX))
    
    #visualize training data
    plt.imshow(train_set_x.get_value()[2000:2001].reshape(patch_size,patch_size),cmap = cm.Greys_r)
    plt.title("train_patch #2000")
    plt.show()


    train_set_y = numpy.loadtxt(train_set_y_path)
    train_set_y = train_set_y[:split_train]
    train_set_y = theano.shared(numpy.asarray(train_set_y,
                                               dtype=theano.config.floatX))

    train_set_y = T.cast(train_set_y, 'int32')
    
    test_o_set_x = numpy.loadtxt(test_set_x_path)    
    test_o_set_x = test_o_set_x[split_test:end_test]
    print ("test_set_o_x.shape : ",test_o_set_x.shape)
    test_o_set_x = theano.shared(numpy.asarray(test_o_set_x,dtype=theano.config.floatX))
    
    
    #here load the real image as testing  235225
    test_set_x = numpy.loadtxt('05_png_test_42.txt')
    print ("test_set_x.shape : ",test_set_x.shape)
    test_set_x = theano.shared(numpy.asarray(test_set_x,
                                               dtype=theano.config.floatX))
    
    plt.imshow(test_set_x.get_value()[2000:2001].reshape(patch_size,patch_size),cmap = cm.Greys_r)
    plt.title("test_patch #2000")
    plt.show()    

    test_set_y = numpy.loadtxt(test_set_y_path)
    print ("test_set_y_test.shape : ",test_set_y.shape)
    test_set_y = test_set_y[split_test:end_test]
    #train_set_y = train_set_y.reshape(20,1)
    #print "read patch :", train_set_x
    
    test_set_y = theano.shared(numpy.asarray(test_set_y,
                                               dtype=theano.config.floatX))
    test_set_y = T.cast(test_set_y, 'int32')
     
    valid_set_x = numpy.loadtxt(test_set_x_path)
    valid_set_x = valid_set_x[split_test:end_test]
    valid_set_x = theano.shared(numpy.asarray(valid_set_x,
                                               dtype=theano.config.floatX))

    valid_set_y = numpy.loadtxt(test_set_y_path)
    valid_set_y = valid_set_y[split_test:end_test]
    valid_set_y = theano.shared(numpy.asarray(valid_set_y,
                                               dtype=theano.config.floatX))

    valid_set_y = T.cast(valid_set_y, 'int32')
    ####
        
    n_train_batches = split_train  / batch_size
    n_valid_batches = (end_test-split_test) / batch_size
    if(test_set_x.shape[0] ==221841):
        print ("test_set_x size is : ",test_set_x.shape[0])
        n_test_batches = (221841) / batch_size
    else:
        n_test_batches = 221841 / batch_size

    n_test_o_batches = (end_test-split_test)/batch_size
    #//todo prepare the testing data (maybe also validation data), so that we can run the whole programm
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, patch_size, patch_size))
    
        
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, patch_size, patch_size),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    #how to print first layer output:
    print ("type of layer0 : ",layer0.output.eval({x:train_set_x.get_value()[:batch_size]}).shape)
    #why here[:50] must be equal to batch size?
    layer0_output = layer0.output.eval({x:train_set_x.get_value()[:batch_size]})
    
    #print( "one sample of layer0 : ",layer0_output[0][1])
    print (layer0_output.shape)
    
    plt.imshow(layer0_output[0][1],cmap = cm.Greys_r)
    plt.title("first layer output, size:12*12")
    plt.show()
    

    #print "outputs of layer0 : ",layer0.output.eval().shape#??i cannot print out the output? how can i checl the output is right?
    #??so output should be 14*14, but see the following image shape, it is 12 *12
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    #theano.printing.Print("layer 0", layer0)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 19, 19),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(3, 3)
    )
    
    print ("x shape :",train_set_x.get_value().shape)
    
    #why the layer1 output is not 1 pixel? but 4*4
    #print ("layer1 output : ",layer1.output.eval({x:train_set_x.get_value()[:batch_size]}))
    layer1_output = layer1.output.eval({x:train_set_x.get_value()[:batch_size]})
    print ("layer1 output shape : ",layer1_output.shape)
    plt.imshow(layer1_output[0][1],cmap = cm.Greys_r)
    plt.title("second layer output, size:4*4")
    plt.show()
    
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 5, 5),
        filter_shape=(nkerns[1], nkerns[1], 5, 5),
        poolsize=(1, 1)
    )
    
    print ("layer2 output : ",layer2.output.eval({x:train_set_x.get_value()[:batch_size]}))
    layer2_output = layer2.output.eval({x:train_set_x.get_value()[:batch_size]})
    print ("layer2 output shape : ",layer2_output.shape)
    plt.imshow(layer1_output[0][1],cmap = cm.Greys_r)
    plt.title("second layer output, size:4*4")
    plt.show()
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer3_input = layer2.output.flatten(2)##??
    
    #print "layer2_input : ",numpy.asarray(layer2_input)

    # construct a fully-connected sigmoidal layer
    #??why here is not 
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[1] * 1 * 1,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    predicated = layer4.y_pred
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_o_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)  

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    prediction = theano.function(
        [index], 
        predicated, 
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }                                                       
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print ('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    #//todo1: using 05_png_patch_test.txt as test_set_x, first see if the data shape is correct, 
    #//todo2: print the predict results, see if the total number of prediction matches input shape
    #//todo3: write the prediction result to file, and reconstruct the imge using predicted labels.
    
    #n_train_batches = 2
    #=====
    inti = numpy.array([])
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print ('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            #print(cost_ij)

            if True:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                '''
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                '''
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    print ("n_test_batches : ",n_test_batches)
                    # test it on the test set
                    
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_o_batches)
                    ]    
                    
                    test_score = numpy.mean(test_losses)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    
                    test_predictions = [prediction(i) for i in xrange(n_test_batches)]
                    print  ("type of prediction : ",type(test_predictions[0]))
                    
                    print ("number of element in prediction :", len(test_predictions))
                    #print "test prediciton : ",test_predictions
                    final = numpy.array([])
                    for each in range(len(test_predictions)):
                        tmp = numpy.concatenate((final,test_predictions[each]))
                        final = tmp
                    #print final
                    open('final_prediction_'+str(minibatch_index)+"_"+str(patch_size)+"_"+str(no_of_layer)+"_Layers2.txt","w")
                    numpy.savetxt('final_prediction_'+str(minibatch_index)+"_"+str(patch_size)+"_"+str(no_of_layer)+"_Layers2.txt",final,newline="\n", fmt = "%f")

            if patience <= iter:
                done_looping = True
                break
    #=====
    
    
    

    end_time = time.clock()
    print('Optimization complete.')
    '''
    ?? why we need validation set?
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    '''
    print ('test file is : ', str(patch_size))
    print('test performance %f %%' %
          ( test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)