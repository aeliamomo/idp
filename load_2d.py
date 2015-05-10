'''
author : chingyu kao
This file deal with 2d image and its labeled file.
load(image)->save as numpy array
load, save and conver_to_itk is used from Markus's file, in medvis folder.
'''

import numpy as np
import os
import SimpleITK as itk
from SimpleITK._SimpleITK import new_AbsImageFilter
from matplotlib import matplotlib_fname
from numpy.random.mtrand import shuffle

try:
    import itk as ITK
except ImportError:
    print 'ITK wrappers not found. Only limited functionality is provided.'
import numpy as np
from medvis import load,save,convert_to_itk
import pylab as plt 
import matplotlib.cm as cm
import math
import matplotlib

def prepare2DLabels(label_file):
    
    #images = load(image_file,convert=True)#convert = Ture means it will return an array
    labels = load(label_file,convert=True)
    
    #print "images'shape : ",images.shape
    print "labels'shape : ",labels.shape #(1,512,512)
    
             # the first figure
    plt.imshow(labels[0,:,:],cmap = cm.Greys_r)
    plt.show()
    labeled_image = labels[0,:,:]     
    return labeled_image

def patchify(img, patch_shape):
    ###################################################
    #                                                 #
    # This method take patches from the whole image   #
    #                                                 #
    ###################################################
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def getPatchAccordingLabel(img,label_image,label, patch_x,patch_y,flatten_as_1D=True):
    ####################################################
    #
    # 
    #
    ####################################################
    print "======take patch around certain pixel (i,j)======"
    print img.shape
    print "label shape",label_image.shape
    
    
    mask_y, mask_x = np.where((label_image == label))
    
    pixels = []
    for col,row in zip(mask_x,mask_y):
        print (row,col)
        pixels.append((row,col))
        
    diff = math.floor(patch_x/(2.0))

    #new_x_array = np.empty((1,patch_x*patch_y))
    new_x_array = []

    count = 0
    for m in range(len(pixels)):
        i = pixels[m][0]
        j = pixels[m][1]
        #print "i",i
        #print "j",j
        
        if(i+diff <= len(label_image) and j+diff<= len(label_image) and i-diff >= 0 and j-diff >= 0):
            if(patch_x % 2 ==0):
                rect = np.copy(img[i-diff:i+diff,j-diff:j+diff])
            else :
                rect = np.copy(img[i-diff:i+diff+1,j-diff:j+diff+1])
            #print rect
            #print "length :",len(rect)
            if(flatten_as_1D == True and rect.shape[0] == patch_x and rect.shape[1] == patch_y):
                count = count + 1 
                rect = rect.reshape(1,patch_x*patch_y)
                #print "rect is : ",rect.shape 
                new_x_array.append(rect)
            
            #return rect
        else : 
           
            print "index exceed the range."
    new_x_array = np.asarray(new_x_array)

    sample_number = new_x_array.shape[0]
    print "number of sample : ",sample_number
    new_x_array_reshape = new_x_array.reshape(1,sample_number*patch_x*patch_y)

    result = new_x_array_reshape.reshape(sample_number,patch_x*patch_y)

    if( label == 1):
        label_tmp = np.ones((sample_number,label))
        result = np.hstack((result,label_tmp))
        print "label ==1 ",result.shape
    if (label == 2): 
        label_tmp = np.zeros((sample_number,label-1))
        result = np.hstack((result,label_tmp))
        
        print "label == 2",result.shape
    
    return result



def main():
    np.set_printoptions(threshold=np.nan)
    
    saved_x = 'patch_train_x_42_shuffled_05'
    saved_y = 'patch_train_y_42_shuffled_05'
    patch_size = 42
       
    print "====Create 2D label data and save as a file====="
    
    label_file= "05-label.mhd"
    labels_2D = prepare2DLabels(label_file)
    print labels_2D.shape
    '''
    open('05_labeled_detail.txt', 'w')
    np.savetxt("05_labeled_detail.txt", labels_2D, newline="\n", fmt = "%d")
    '''
    '''
    read png image
    '''
    file = '05.png'
    im = matplotlib.image.imread(file)
    
    '''
    create patches using image and labeled files
    '''
    patch_label_one = getPatchAccordingLabel(im, labels_2D, label = 1, patch_x = patch_size, patch_y = patch_size)   
    patch_label_zero = getPatchAccordingLabel(im, labels_2D, label = 2, patch_x = patch_size, patch_y = patch_size)
   
    print patch_label_one.shape
    print patch_label_zero.shape
    
    '''
    open('05_label_one_only_detail.txt','w')
    np.savetxt('05_label_one_only_detail.txt',patch_label_one,newline="\n",fmt="%f")
    open('05_label_zero_only_detail.txt','w')
    np.savetxt('05_label_zero_only_detail.txt',patch_label_zero,newline="\n",fmt="%f")
    '''
    
    '''
    parameter: sample_of_label_one : count how many sample labeled one is 
    in order to take same amount sample from positive and negative labeled pixels
    '''
    sample_of_label_one = patch_label_one.shape[0]
    print "sampel of label one : ",sample_of_label_one
    #take same amout of label one and label two 
    
    '''merge two samples'''
    merged = np.vstack((patch_label_one,patch_label_zero[:sample_of_label_one]))
    np.random.shuffle(merged)
    
    #open("train_patch_set","w")
    #np.savetxt('train_patch_set',merged,newline="\n", fmt = "%f")
    
    '''shuffle the merged samples and save as new files'''
    train_x_shuffled = merged[:,:patch_size*patch_size]
    print train_x_shuffled.shape
    train_y_shuffled = merged[:,patch_size*patch_size:patch_size*patch_size+1]
    print train_y_shuffled.shape
    
    open(saved_x, 'w')
    np.savetxt(saved_x, train_x_shuffled, newline="\n", fmt = "%f")

    open(saved_y, 'w')
    np.savetxt(saved_y, train_y_shuffled, newline="\n", fmt = "%d")
    
    
    
    print "===read png as patches==="
    im = matplotlib.image.imread("05.png")

    result = patchify(im, (patch_size,patch_size))
    print "result shape : ",result.shape

    png_array = []
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
        #print result[i,j,:,:]
            final = result[i,j,:,:].reshape(1,patch_size*patch_size)
            png_array.append(final)
    png_array = np.asarray(png_array)
    print png_array.shape
    #print png_array[0][0]

    tmp = []
    for each in range(png_array.shape[0]):   
        tmp.append(png_array[each][0])

    tmp = np.asarray(tmp)
    print tmp.shape

    open('05_png_test_42.txt','w')
    np.savetxt('05_png_test_42.txt',tmp,newline="\n", fmt = "%f")

    

if __name__ == '__main__':
    main()
