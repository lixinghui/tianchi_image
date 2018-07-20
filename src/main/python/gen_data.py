from random_cut import random_cut
import pandas as pd
import scipy.misc
from tqdm import tqdm
def gen_data(num_each=50):
	df_label=pd.read_csv("../../../data/xls/labels.csv")
	for i in tqdm(range(df_label.shape[0])):
		for j in range(num_each):
			filename=df_label.loc[i,'filename']
			x_min=df_label.loc[i,'xmin']
			y_min=df_label.loc[i,'ymin']
			x_max=df_label.loc[i,'xmax']
			y_max=df_label.loc[i,'ymax']
			image_cut=random_cut(filename,x_min,y_min,x_max,y_max)
			scipy.misc.imsave('../../../data/xls/data_gen/{1}_{0}'.format(filename,j),image_cut)
		
gen_data()
