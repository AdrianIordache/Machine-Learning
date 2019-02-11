
# coding: utf-8

# In[12]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/Adi/Desktop/Deep Learning/Dog or Cats Net/Dataset/PetImages"

CATEGORIES = ['Dog', 'Cat']

for category in CATEGORIES:
    #path to cats or dogs folder
    path = os.path.join(data_dir, category)
    for image in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = 'gray')
        plt.show()
        break
    break


# In[7]:


print(img_array)


# In[10]:


IMAGE_SIZE = 100
new_images = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
plt.imshow(new_images, cmap = 'gray')
plt.show()


# In[14]:


training_data = []

DATA_DIR = "C:/Users/Adi/Desktop/Deep Learning/Dog or Cats Net/Dataset/PetImages"

CATEGORIES = ['Dog', 'Cat']

def create_training_data():
    for category in CATEGORIES:
        #path to cats or dogs folder
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                new_images = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_images, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))


# In[15]:


import random
random.shuffle(training_data)
X = []
y = []
for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


# In[16]:


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[18]:


pickle_in = open("X.pickle", "rb")
x = pickle.load(pickle_in)


# In[19]:


X[1]

