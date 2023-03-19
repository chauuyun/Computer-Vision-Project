import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import math
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Activation

PATH=os.getcwd()
train_file = 'train_list.txt'
test_file = 'test_list.txt'
print("PATH:",PATH)
def get_data_from_file(train_file):
  train_data = []
  name_list=[]
  img_list=[]
  num_list=[]
  test_data=[]
  Y_train=[]
  with open(train_file) as fp:
     for lines in fp:
        # obtain img_path from line
        lines = fp.readlines()
        for line in lines:
            img_path=line.strip().split('/')
            name_list.append(img_path[0:1][0])
            img_num=img_path[1:]
            img_list.append(img_num[0])
            for line in img_num:
                num= line.strip('.jpg').split()
                integer = np.array( num, dtype="int32")
                num_list.append(integer[0])
     j=len(num_list)
     X=np.zeros([j,48, 48,3])
     q=-1
     for i in range(0,j): 
         # print(name_list[i])
         path1 = os.path.join(PATH, name_list[i])
         path2 = os.path.join(path1, img_list[i])
         # print(path2)
         
         try:
            img = cv2.imread(path2)
            img_resized = cv2.resize(img, (48, 48))

            if "Cat" in name_list[i]:
               label = 0
               q=q+1
            if "Dog" in name_list[i]:
               label = 1
               q=q+1
            if (label==0)or(label==1):
               X[q:q+1,:,:,:]=img_resized
               train_data.append([label,img_resized])
            Y_train.append(label)
         except:
              print (i,"error")
           
  return train_data,X,Y_train


def get_image_and_label(train_data,X,Y_train):
  num_X=len(train_data)
  X_train=np.zeros([num_X,48, 48,3])
  X_train=X[0:num_X,:,:,:]   
  Y_train=np.array(Y_train)
  yy=X_train[400]
  Y_train=np.array(Y_train)
  return X_train,Y_train

def assignment3b_1():
  train_file = 'train_list.txt'
  test_file = 'test_list.txt'
  train_data,X,Y_train = get_data_from_file(train_file)
  X_train, Y_train = get_image_and_label(train_data,X,Y_train)
  test_data,X,Y_test=get_data_from_file(test_file)
  X_test, Y_test=get_image_and_label(test_data,X,Y_test)
  save_path = 'dogs_cats.pkl'
  print('Saving to', save_path)
  data = {}
  data['X_train'] = X_train
  data['Y_train'] = Y_train
  data['X_test'] = X_test
  data['Y_test'] = Y_test
  pickle.dump(data, open(save_path, 'wb'))
  
def assignment3b_2():
  data = pickle.load(open('dogs_cats.pkl','rb'))
  
  X_train, Y_train, X_test, Y_test = data["X_train"], data["Y_train"], data["X_test"], data["Y_test"]
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  model = keras.Sequential([
  keras.layers.Flatten(input_shape=(48,48,3)),
  keras.layers.Dense(256, activation=tf.nn.relu),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(2, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])
  tb_callback = TensorBoard(log_dir='log_fashion')
  checkpoint_path="ckpt_fashion/cp-{epoch:04d}.ckpt"
  cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)
  model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=2, callbacks=[cp_callback, tb_callback])
  
  for epoch in [5, 10, 15, 20]:
      latest = checkpoint_path.format(epoch=epoch)
      model.load_weights(latest)
  
  weight_path = "ckpt_fashion/cp-0020.ckpt"
  model.load_weights(weight_path)
  test_loss1, test_acc1 = model.evaluate(X_test, Y_test)
  train_loss1, train_acc1 = model.evaluate(X_train, Y_train)
  print("'sparse_categorical_crossentropy'_test_acc:",test_acc1)
  print("'sparse_categorical_crossentropy'_train_acc:",train_acc1)

def assignment3b_3():
  data = pickle.load(open('dogs_cats.pkl','rb'))
  X_train, Y_train, X_test, Y_test = data["X_train"], data["Y_train"], data["X_test"], data["Y_test"]
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  model = keras.Sequential([
  keras.layers.Flatten(input_shape=(48,48,3)),
  keras.layers.Dense(256, activation=tf.nn.relu),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(2, activation="linear")
  ])
  model.compile(optimizer='adam',
                 loss='mean_squared_error',
                 metrics=['accuracy'])
  tb_callback = TensorBoard(log_dir='log_fashion')
  checkpoint_path="ckpt_fashion/cp-{epoch:04d}.ckpt"
  cp_callback = ModelCheckpoint(checkpoint_path, 
                                save_weights_only=True, 
                                verbose=1, period=5)
  model.fit(X_train, Y_train, batch_size=32, 
            epochs=20, verbose=2, 
            callbacks=[cp_callback, tb_callback])
  
  for epoch in [5, 10, 15, 20]:
      latest = checkpoint_path.format(epoch=epoch)
      model.load_weights(latest)
  
  weight_path = "ckpt_fashion/cp-0020.ckpt"
  model.load_weights(weight_path)
  test_loss2, test_acc2 = model.evaluate(X_test, Y_test)
  train_loss2, train_acc2 = model.evaluate(X_train, Y_train)
  print("'mean_squared_error'_test_acc:",test_acc2)
  print("'mean_squared_error'_train_acc:",train_acc2)
  reg2 = model.predict(X_test).squeeze()
  predictions=[]
  reg=[]
  reg=np.argmax(reg2,axis=1)
  print("reg.shape:",reg.shape)   
  for i in range(len(reg)):
       if reg[i]>0.5 :
         pre_acc=1
       if reg[i]<0.5 :
         pre_acc=0
       predictions.append(pre_acc)
  
  pred_res = (predictions == Y_test)
  true_num = np.sum(pred_res!=0)
  total_number = len(pred_res)
  test_accuracy = math.floor((true_num/total_number) * 1000) / 1000.0
  test_acc2=math.floor(test_acc2 * 1000) / 1000.0
  if (test_accuracy==test_acc2):
      print("The accuracy is the same!")
  else:
      print("The accuracy is not the same...") 
      
data_dir = 'PetImages'
img_size = 48

if __name__ == '__main__':
    assignment3b_1()
    assignment3b_2()
    assignment3b_3()

##Some discussions:
# Question 3b-2
# The test accuracy is much lower than the training accuracy. Why?
# How can we improve the test accuracy?.
#因為Training的data較多，模型訓練較充分。

#如何提高test accuracy:
#1.將偏差數據或遺失的數據剔除
#2.直接從原始數據中提取特徵(Feature Selection or Feature Engineering)，會使得機器學習的結果質量更高，test accuray進而提升
#3.可以使用不同的演算法，或是將更新式做優化
#4.可以調整超參數，像是更換loss function的種類...等等

# Question 3b-3
# Compare the test accuracy of mean_squared_error to that of sparse_categorical_crossentropy. 
# Which one is better? Why?
# Write down your answer here.

#sparse_categorical_crossentropy的test accuracy較高
#sparse_categorical_crossentropy的標籤是使用0與1格式（One-hot 編碼形式）
#mean_squared_error則是使用最小二乘法去求得，數學定義則是預測向量與真實向量差值的平方然後求平均
#sparse_categorical_crossentropy之所以準確率較高是因為它將標籤改成0與1的形式去做預測
#而在此程式中就是將貓或狗的照片種類區分為0或1，自然訓練出來的準確率就會較高
#而在梯度下時，Cross Entropy計算速度較MSE快，所以很多訓練模型都會選擇使用Cross Entropy種類的loss function
