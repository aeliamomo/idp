import numpy as np
import pylab as plt
import matplotlib.cm as cm
import math



prediction_file = np.loadtxt('final_prediction_0_42_3_Layers2.txt')
row = prediction_file.shape[0]
print ("row",row)
shape = prediction_file.shape
print ("shape",shape)
reshape_size = math.sqrt(row)
print ("reshape size is before int : ",reshape_size)
reshape_size = int(math.sqrt(row))
print ("reshape size is : ",reshape_size)
prediction_file = prediction_file.reshape(reshape_size,reshape_size)
print prediction_file.shape
plt.imshow(prediction_file,cmap = cm.Greys_r)
plt.show()

