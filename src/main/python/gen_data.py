from random_cut import random_cut
import numpy as np
import pandas as pd
import os
import scipy.misc
from tqdm import tqdm
def gen_flaw_data(num_each=50,shape=448):
	df_label=pd.read_csv("../../../data/xls/labels.csv")
	for i in tqdm(range(df_label.shape[0])):
		if i<100:
			for j in range(num_each):
				filename=df_label.loc[i,'filename']
				x_min=df_label.loc[i,'xmin']
				y_min=df_label.loc[i,'ymin']
				x_max=df_label.loc[i,'xmax']
				y_max=df_label.loc[i,'ymax']
				image_cut=random_cut(filename,x_min,y_min,x_max,y_max)
				scipy.misc.imsave('../../../data/xls/val/flaw/{}_{}_{}_{}{}'.format(filename[:-4],
								x_min,y_min,j,filename[-4:]),image_cut)
		else:
			for j in range(num_each):
				filename=df_label.loc[i,'filename']
				x_min=df_label.loc[i,'xmin']
				y_min=df_label.loc[i,'ymin']
				x_max=df_label.loc[i,'xmax']
				y_max=df_label.loc[i,'ymax']
				image_cut=random_cut(filename,x_min,y_min,x_max,y_max,shape=shape)
				scipy.misc.imsave('../../../data/xls/train/flaw/{}_{}_{}_{}{}'.format(filename[:-4],
								x_min,y_min,j,filename[-4:]),image_cut)
		
def gen_normal_data(num_each=50,shape=448):
	files=os.listdir("../../../data/xls/normal/")
	for i,filename in tqdm(enumerate(files)):
		image_data=scipy.misc.imread("../../../data/xls/normal/{}".format(filename))
		x_lower=0
		x_upper=2560-shape
		y_lower=0
		y_upper=1920-shape
		for j in range(num_each):
			x=np.random.randint(x_lower,x_upper)
			y=np.random.randint(y_lower,y_upper)
			image_cut=image_data[y:y+shape,x:x+shape]
			if i <100:
				scipy.misc.imsave('../../../data/xls/val/normal/{}_{}{}'.format(filename[:-4],j,filename[-4:]),image_cut)
			else:
				scipy.misc.imsave('../../../data/xls/train/normal/{}_{}{}'.format(filename[:-4],j,filename[-4:]),image_cut)
		
		
gen_flaw_data()
gen_normal_data()
