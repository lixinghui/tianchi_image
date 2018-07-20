import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_cut(filename,x_min,y_min,x_max,y_max,shape=448):
	x_c=int((x_min+x_max)/2)
	y_c=int((y_min+y_max)/2)
	if x_max-x_min <=shape:
		x_lower=max(x_c-shape,0)
		x_upper=min(x_c+shape,2560)-shape
	else:
		x_lower=max(x_min-shape/2,0)
		x_upper=min(x_max+shape/2,2560)-shape
	if y_max-y_min<=shape:
		y_lower=max(y_c-shape,0)
		y_upper=min(y_c+shape,1920)-shape
	else:
		y_lower=max(y_min-shape/2,0)
		y_upper=min(y_max+shape/2,1920)-shape

	x=np.random.randint(x_lower,x_upper)
	y=np.random.randint(y_lower,y_upper)
	image=scipy.misc.imread('../../../data/xls/flaw/{}'.format(filename))
	image_cut=image[y:y+shape,x:x+shape,:]
	
	return image_cut
	
#plt.figure(figsize=(12,12))
#for i in range(9):
#	plt.subplot(3,3,i+1)
#	image_cut=random_cut("J01_2018.06.17 13_33_05.jpg",1109,262,1677,1022)
#	plt.imshow(image_cut)
#
#plt.show()
