import pandas
import numpy as np


import csv
f = csv.reader(open('L:\\shangdong\\2022\\2\\train2_new.csv','r'))

json_down = {}

muti={
'ShortSleeve':0,
'LongSleeve':1,
'NoSleeve':2,
'multicolour':3,
'Solidcolor':4,
'lattice':5,
'Short':6,
'Long':7,
'middle':8,
'Bald':9,
'Skirt':10,
'Trousers':11,
'Shorts':12,
'multicolour2':13,
'Solidcolor2':14,
'lattice2':15,
'else':16,
'LeatherShoes':17,
'Sneaker':18,
'Sandals':19,
'right':20,
'back':21,
'front':22,
'left':23
}
# upperBlack	
# upperBrown	
# upperBlue	
# upperGreen	
# upperGray	

# upperOrange	
# upperPink	
# upperPurple	
# upperRed	
# upperWhite	

# upperYellow	
# lowerBlack	
# lowerBrown	
# lowerBlue	
# lowerGreen	

# lowerGray	
# lowerOrange	
# lowerPink	
# lowerPurple	
# lowerRed	

# lowerWhite	
# lowerYellow

number = 46

for i,row in enumerate(f):
	if i==0:
		continue
	else:
		one_label = np.zeros(number)
		name = row[0]
		for j in range(1,len(row)):
			if j<=7:
				if j!=5:
					select = row[j]
					index = muti[select]
					one_label[index]=1
				else:
					select = row[j]+'2'
					index = muti[select]
					one_label[index]=1
			else:
				if row[j]!='':
					# print(row[j])
					float_label = float(row[j])
					one_label[j+16]=float_label
		json_down[name]=one_label
	# print(json_down)
	# break
	if i%500==0:
		print(i)

np.save("L:\\shangdong\\2022\\2\\label.npy",json_down)

from sklearn.model_selection import train_test_split

# label_npy = np.load("L:\\shangdong\\2022\\2\\label.npy",allow_pickle=True).item()
# # print(label_npy.keys())
# image_name_list = []
# label_list =[]

# sta = np.zeros((4000,21))
# i=0
# for key_name in label_npy.keys():
# 	image_name_list.append(key_name)
# 	label_list.append(label_npy[key_name])
# 	sta[i] = np.array(label_npy[key_name])
# 	i+=1


# gt_pos = np.sum((sta>0),axis=0).astype(float)


# print(gt_pos)


# train_data,test_data,train_target,test_target = train_test_split(image_name_list, label_list, test_size = 0.25,random_state=0)

# train_target = np.array(train_target)
# gt_pos = np.sum((train_target>0),axis=0).astype(float)


# print(gt_pos)
# print(len(train_data),train_data[0])
# print(len(test_data),test_data[0])
# label_list = np.concatenate(label_list)






# from PIL import Image
# import os 

# # root = 'L:\\shangdong\\2022\\testA'

# root = 'L:\\shangdong\\2022\\train\\train1A'

# file = os.listdir(root)

# import numpy as np

# from matplotlib import pyplot as plt 
# import numpy as np  




# height = []
# weight = []


# for name in file:
#     path = os.path.join(root,name)
#     img = Image.open(path)
#     height.append(img.size[1])
#     weight.append(img.size[0])

# height = np.array(height)
# weight = np.array(weight)

# # print(np.mean(height))
# # print(np.mean(weight))

# # print(weight)
# # print(height)
# # a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
# print([x for x in range(100,300,20)])
# # plt.hist(height, bins =  [x for x in range(100,300,20)]) 
# plt.hist(weight, bins =  [x for x in range(100,300,10)]) 

# plt.title("histogram") 
# plt.show()