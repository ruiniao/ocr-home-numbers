#!/usr/bin/env python
# coding: utf-8

# ## Table of contents:
# * [Project description: Build a simple OCR system](#first-bullet)
# * [Solution pipeline](#second-bullet)
#   * [1. Explore the dataset](#second-first-bullet)
#   * [2. Data preprocessing and formatting](#second-second-bullet)
#   * [3. Train/Val/Test dataset](#second-third-bullet)
#   * [4. Build models](#second-fourth-bullet)
#   * [5. Train/evaluate the model](#second-fifth-bullet)
#   * [6. Text recognizer: Ready to predict new images](#second-sixth-bullet)
# * [Recommendations for future work](#third-bullet)

# # Project description: Build a simple OCR system <a class="anchor" id="first-bullet"></a>
# 
# In the `ocr.tar.gz` file there are a number of jpg files that are named as
# 
#     {counter}_{text}.jpg
# 
# where `counter` is just an arbitrary number and `text` represents the characters inside
# the image, sorted left to right. The to-do list of this project is:
# 
# 1.  Build a simple OCR model for this data and train it
# 2.  Write a class that loads up a trained model and has a `predict` function which should accept an image as an input and return the OCR result as a string
# 3.  Show the loss curves as well as some accuracy tests with examples
# 4.  Discuss the recommendations for improvements

# # Solution pipeline <a class="anchor" id="second-bullet"></a>
# Since the dataset contains focused images with the digit sequence as the main object, here we will not do text region detection. We will use the image data as the input of a multi-output classifier to classify the digits in the image. Note that the maximum length of digits in the image is 5. The base model is a CNN model with 5 convolutional layers and 5 dense output layers.
# 

# In[104]:


#if use google colab, uncomment this.
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# !ls "/content/drive/My Drive/Colab Notebooks"
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/ocr/')


# # 1. Explore the dataset <a class="anchor" id="second-first-bullet"></a>
# * Check samples images.
# * Check label distribution.

# In[ ]:


import tarfile
import os
import glob
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs={'bbox_inches':None}")
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
sns.set(rc={'figure.figsize':(20, 10)})
import warnings
warnings.filterwarnings('ignore')

from os.path import join
from skimage.io import imread
from skimage.color import rgb2gray
from collections import Counter
from numba import jit
from cv2 import resize
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D


# In[ ]:


#unpack the dataset
data_file = tarfile.open("/content/drive/My Drive/Colab Notebooks/ocr/ocr.tar.gz")
data_file.extractall("/content/drive/My Drive/Colab Notebooks/ocr/")
data_file.close()


# In[94]:


#get the data file list
@jit
def get_img_list(data_path):
    return os.listdir(data_path)
data_path = "/content/drive/My Drive/Colab Notebooks/ocr/ocr_dataset/"
img_files = get_img_list(data_path)
print(len(img_files))
img_files[:5]


# In[ ]:


@jit
def visualize(data_path, img_files, n_cols = 5, n_rows=2):
    plt.figure(figsize = (2*n_cols,2*n_rows))
    for n,i in enumerate(np.random.choice(range(len(img_files)), size = n_cols*n_rows)):
        plt.subplot(n_rows,n_cols,n+1)
        plt.axis('off')
        plt.imshow(imread(join(data_path, img_files[i]))) #or .resize(32,32)
        plt.title(img_files[i].split('.')[0].split('_')[-1])
    plt.show()


# In[96]:


#Visualize sample images
visualize(data_path, img_files, n_cols = 5, n_rows=2)


# In[97]:


#Creat ground truth dataframe
df = []
for i in range(len(img_files)):
    df.append([img_files[i], img_files[i].split('.')[0].split('_')[-1]])
df = pd.DataFrame(df)
df.columns = ['img_name', 'label']
df.head()
print('Numbers of digits:', set([len(ele) for ele in df['label']]))
print('Maximum number of digits:', max(set([len(ele) for ele in df['label']])))


# In[98]:


#Plot number of digits in the dataset
sns.set_context("talk", font_scale=1.4)
number_counter = dict(Counter([len(ele) for ele in df['label']]))
number_digits = pd.DataFrame.from_dict(number_counter, orient='index')
number_digits.columns = ['count']
sns.barplot(x ='index', y='count', 
            data=number_digits.reset_index());
plt.xlabel('length')
plt.title('Distribution of the number of digits appeared in the dataset')


# In[99]:


#Create the counters for 10 digits.
digit_counter = dict({str(i):0 for i in range(10)})
print('Initial digit counter is:',digit_counter)
for i in range(len(df['label'])):
    temp_counter = dict(Counter(df['label'][i]))
    for key in temp_counter:
      temp = digit_counter[key]
      digit_counter[key] = temp + temp_counter[key]    
print('Final digit counter is:', digit_counter)


# In[100]:


#Plot digit distribution in the dataset
sns.set_context("talk", font_scale=1.4)
digit_count = pd.DataFrame.from_dict(digit_counter, orient='index')
digit_count.columns = ['count']
sns.barplot(x ='index', y='count', 
            data=digit_count.reset_index());
plt.xlabel('digit')
plt.title('Distribution of the count of the digits appeared in the dataset')


# # 2. Data preprocessing and formatting <a class="anchor" id="second-second-bullet"></a>
# Please note that image data reading and preprocessing is much faster using local machine than using colab and Google Drive. Therefore, this section can be done at local machine. And remember to save the processed data.

# In[101]:


#label preprocessing: padding the labels with a fixed length of 5
df[['0','1','2','3','4']] = df['label'].apply(lambda x: pd.Series([10]*(5-len(x))+[int(i) for i in x])) #list(map(lambda x: [10]*(5-len(x))+[int(i) for i in x], df['label']))
df.head()
# df.iloc[0]


# In[ ]:


#load image data and resize images
@jit()
def load_image_data(data_path, data):
    images, labels = [], []
    n = len(data)    
    for i in range(n):
        if i%1000 == 0:
            print(i, data.iloc[i,0])        
        img = imread(join(data_path, data.iloc[i,0]))
        img = rgb2gray(img)
        img = resize(img, (32,32))
        images.append(img)
        labels.append(data.iloc[i,2:].as_matrix().astype('int16'))
    return images, labels
all_images, all_labels = load_image_data(data_path, df)


# In[ ]:


with open('/content/drive/My Drive/Colab Notebooks/ocr/all_images.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(all_images, filehandle)

with open('/content/drive/My Drive/Colab Notebooks/ocr/all_labels.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(all_labels, filehandle)


# # 3. Train/Val/Test dataset <a class="anchor" id="second-third-bullet"></a>
# Prepare the datasets for training/validation/testing. The ratio we used is 4:1:1 for train vs. val vs. test.

# In[ ]:


#Load preprocessed data
with open('/content/drive/My Drive/Colab Notebooks/ocr/all_images.data', 'rb') as filehandle:
    # read the data as binary data stream
    all_images = pickle.load(filehandle)

with open('/content/drive/My Drive/Colab Notebooks/ocr/all_labels.data', 'rb') as filehandle:
    # read the data as binary data stream
    all_labels = pickle.load(filehandle)


# In[106]:


print('Label: ', all_labels[10])
plt.figure(figsize=(2,2))
plt.imshow(all_images[10], cmap=plt.cm.gray);


# In[107]:


labels = pd.DataFrame(list(map(np.ravel, all_labels)))
labels.head()


# In[108]:


images = pd.DataFrame(list(map(np.ravel, all_images)))
images.head()


# In[109]:


@jit
def digit_to_categorical(data):
    n = data.shape[1]
    data_cat = np.empty([len(data), n, 11])    
    for i in range(n):
        data_cat[:, i] = to_categorical(data[:, i], num_classes=11)        
    return data_cat
#test
test = digit_to_categorical(labels.iloc[:,:].as_matrix().astype('int16'))
test[0]


# In[ ]:


#Split train/val/test dataset
X = images.values.reshape(-1, 32, 32, 1)
y = digit_to_categorical(labels.iloc[:,:].as_matrix().astype('int16'))

X_0, X_test, y_0, y_test = train_test_split(X, y, test_size=1/6, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_0, y_0, test_size=0.2, random_state=42)
del X,y,X_0,y_0


# In[112]:


X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape


# In[ ]:


y_train_list = [y_train[:, i] for i in range(5)]
y_test_list = [y_test[:, i] for i in range(5)]
y_valid_list = [y_val[:, i] for i in range(5)]


# # 4. Build models <a class="anchor" id="second-fourth-bullet"></a>

# In[ ]:


#define a cnn model
def cnn_model():    
    model_input = Input(shape=(32, 32, 1))
    x = BatchNormalization()(model_input)
        
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)       
    x = Conv2D(64, (3, 3), activation='relu')(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(196, (3, 3), activation='relu')(x)    
    x = Dropout(0.25)(x)
              
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.5)(x)
    
    y1 = Dense(11, activation='softmax')(x)
    y2 = Dense(11, activation='softmax')(x)
    y3 = Dense(11, activation='softmax')(x)
    y4 = Dense(11, activation='softmax')(x)
    y5 = Dense(11, activation='softmax')(x)
    
    model = Model(input=model_input, output=[y1, y2, y3, y4, y5])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# # 5. Train/evaluate the model <a class="anchor" id="second-fifth-bullet"></a>

# In[13]:


#Trainig a basic model
cnn_model = cnn_model()
cnn_checkpointer = ModelCheckpoint(filepath='/content/drive/My Drive/Colab Notebooks/ocr/models/weights_best_cnn.hdf5', verbose=2, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
cnn_history = cnn_model.fit(X_train, y_train_list, 
                            validation_data=(X_val, y_valid_list), 
                            epochs=4000, batch_size=128, verbose=2, 
                            callbacks=[es, cnn_checkpointer])


# In[115]:


#Evaluate
cnn_model = cnn_model()
cnn_model.load_weights('/content/drive/My Drive/Colab Notebooks/ocr/models/weights_best_cnn.hdf5')
cnn_scores = cnn_model.evaluate(X_test, y_test_list, verbose=0)

print("CNN Model 1. \n")
print("Scores: \n" , (cnn_scores))
print("First digit. Accuracy: %.2f%%" % (cnn_scores[6]*100))
print("Second digit. Accuracy: %.2f%%" % (cnn_scores[7]*100))
print("Third digit. Accuracy: %.2f%%" % (cnn_scores[8]*100))
print("Fourth digit. Accuracy: %.2f%%" % (cnn_scores[9]*100))
print("Fifth digit. Accuracy: %.2f%%" % (cnn_scores[10]*100))

print(cnn_model.summary())


# In[117]:


#predict() test
res = cnn_model.predict(X_test[:1], verbose=1)
res


# In[118]:


#Plotting loss curve
plt.figure(figsize=(14, 7))
plt.plot(cnn_history.history['loss'][:], label = 'Train loss')
plt.plot(cnn_history.history['val_loss'][:], label = 'Validation loss')

plt.legend()
plt.xlabel('epoch')
plt.title('Loss');


# In[119]:


#Plotting accuracy
plt.figure(figsize=(14, 7))

plt.plot(cnn_history.history['val_dense_2_acc'][:], label = 'First digit')
plt.plot(cnn_history.history['val_dense_3_acc'][:], label = 'Second digit')
plt.plot(cnn_history.history['val_dense_4_acc'][:], label = 'Third digit')
plt.plot(cnn_history.history['val_dense_5_acc'][:], label = 'Fourth digit')
plt.plot(cnn_history.history['val_dense_6_acc'][:], label = 'Fifth digit')

plt.legend()
plt.title('Accuracy');


# In[121]:


avg_accuracy = sum([cnn_scores[i] for i in range(6, 11)])/5
print("CNN Model. Average Accuracy for single digit: %.2f%%" % (avg_accuracy*100))

len_1_accuracy = np.prod(cnn_scores[10])
print("CNN Model. Accuracy for 1 digit sequence: %.2f%%" % (len_1_accuracy*100))

len_2_accuracy = np.prod(cnn_scores[9:11])
print("CNN Model. Accuracy for 2 digits sequence: %.2f%%" % (len_2_accuracy*100))

len_3_accuracy = np.prod(cnn_scores[8:11])
print("CNN Model. Accuracy for 3 digits sequence: %.2f%%" % (len_3_accuracy*100))

len_4_accuracy = np.prod(cnn_scores[7:11])
print("CNN Model. Accuracy for 4 digits sequence: %.2f%%" % (len_4_accuracy*100))

total_accuracy = np.prod([cnn_scores[6:11]])
print("CNN Model. Accuracy for 5 digits sequence: %.2f%%" % (total_accuracy*100))


# # 6. Text recognizer: Ready to predict new images <a class="anchor" id="second-sixth-bullet"></a>

# In[ ]:


#define the Recognizer class
class Recognizer():
    def __init__(self):
        self.model = cnn_model()
    
    def predict(self, img_file, model_path):
        self.model.load_weights(model_path)
        img = imread(img_file)
        img = rgb2gray(img)
        img = resize(img, (32,32))
        img = img.reshape(-1, 32, 32, 1)
        pred = self.model.predict(img)
        text = self.decode_text(pred)
        return text
      
    def decode_text(self,res_vec, num_digits=5, num_classes=11, dummy_class=10):
        actual_digits = np.concatenate([np.argmax(res_vec[i],1) for i in range(num_digits)])
        res = actual_digits[actual_digits!=dummy_class]
        return ''.join(map(str, res))


# In[128]:


#test
@jit
def main():
    #initialization
    model_path = '/content/drive/My Drive/Colab Notebooks/ocr/models/weights_best_cnn.hdf5'    
    data_path = "/content/drive/My Drive/Colab Notebooks/ocr/ocr_dataset/"
    img_files = get_img_list(data_path)
    n_cols, n_rows = 5, 2
    test_img_index = np.random.choice(range(len(img_files)), size = n_cols*n_rows)
    plt.figure(figsize = (3*n_cols,3*n_rows))
    ocr = Recognizer()
    
    #get results and plot
    for n,i in enumerate(test_img_index):
        res = ocr.predict(join(data_path,img_files[i]), model_path)
        print('Actual digits are: ', img_files[i].split('_')[-1].split('.')[0], end='-> ')
        print('Predicted digits are: ', res)
        #plot
        plt.subplot(n_rows,n_cols,n+1)
        plt.axis('off')
        plt.imshow(imread(join(data_path,img_files[i]))) #or .resize(32,32)
        plt.title('Predict: '+res)
    plt.show()

main()


# # Recommendations for future work <a class="anchor" id="third-bullet"></a>
# * Data augmentation. Since there are 5567 tesing images not used for training, we could use data augmentation on the training set to significantly increase the diversity of data and avoid overfitting.
# * Progressive resizing. This technique can improve the ability of the model to learn “scale-dependent” patterns. 
# * Attention LSTM. Here we didn't try to use LSTM sequential model because of the amount of data synthesis/augmentation work.

# In[ ]:




