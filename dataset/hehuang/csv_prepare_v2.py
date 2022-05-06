# import pandas
import numpy as np


import csv
# f = csv.reader(open('/media/ubuntu/data/hehuang/car/train/train_sorted.csv','r'))

# json_down = {}

# muti={
# 'van':0,
# 'car':1,
# 'truck':2,
# 'suv':3,
# }


# number = 4

# for i,row in enumerate(f):
# 	if i==0:
# 		continue
# 	else:
# 		one_label = np.zeros(number)
# 		# print(one_label)
# 		# print(row[5])
# 		name = row[0]
# 		for j in range(1,len(row)):
# 			select = row[j]
# 			index = muti[select]
# 			one_label[index]=1
# 			# else:
# 			# 	if row[j]!='':
# 			# 		# print(row[j])
# 			# 		# float_label = float(row[j])
# 			# 		# one_label[j+6]=float_label
# 			# 		one_label[j+6]=1.

# 		json_down[name]=one_label
# 	# print(json_down)
# 	# break
# 	if i%500==0:
# 		print(i)

# np.save("/media/ubuntu/data/hehuang/car/train/label.npy",json_down)

# from sklearn.model_selection import train_test_split

# label_npy = np.load("/media/ubuntu/data/hehuang/car/train/label.npy",allow_pickle=True).item()
# # print(label_npy.keys())
# image_name_list = []
# label_list =[]
# for key_name in label_npy.keys():
# 	image_name_list.append(key_name)
# 	label_list.append(label_npy[key_name])

# train_data,test_data,train_target,test_target = train_test_split(image_name_list, label_list, test_size = 0.25,random_state=0)

# print(len(train_data),train_data[0])
# print(len(test_data),test_data[0])


