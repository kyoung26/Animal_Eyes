import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
import keras
from keras.api.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.api.models import Sequential, load_model
from keras.api.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.api.preprocessing import image

### IMPORTING DATA
DATADIR = r"C:\Users\Keith Young\Desktop\New_dataset" #Directory to my dataset of multiple animal eyes
CATEGORIES = ["Cat", "crocodilians", "Dog", "elephant", "goat", "horse", "lion", "lizard", "anura"] # The names of my folders so that the create_training_data can automatically label them through the loop. Since 


IMG_SIZE = 50  #The size I want the images to print out as
###LABELING DATA
### Source
# https://youtu.be/j-3vuBynnOE?si=Rb0Qcpb-OsQJWC8n
training_data = [] #Will append the trainind data into a list
def create_training_data(): #Method that categorizes my dataset and resizes the images and chages them to grayscale
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            if img.endswith(".jpg"):
                try:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                 pass
          
          
create_training_data()

print(len(training_data))


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #Converts the images into a numpy array of pixel values

y = np.array(y)
y = to_categorical(y, num_classes= len(CATEGORIES))
X = X/255.0
print("Shape of training data images: ", X.shape)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 42)
#idx = random.randint(0, len(X))
#plt.imshow(X[idx, :])
#plt.show()

### MODEL
## My code
## Uses Keras API
# 32 filters
# 3 by 3 kernels
# 1 by 1 stride
# relu activation: makes kernel values if negative return 0 and if positive return 1
model = keras.Sequential()
# first sequence convolutional layers
# max pool 3 by 3
# 
model.add(Conv2D(32, (3,3), activation='relu', input_shape = (IMG_SIZE,IMG_SIZE,1)))
model.add(MaxPool2D((3,3))),
# second sequence of convolutional layers
model.add(Conv2D(32, (3,3), activation='relu', input_shape = (IMG_SIZE,IMG_SIZE,1) ))
model.add(MaxPool2D((3,3)))
# Fully Connected layer
# reLu
# softmax gives probably of features
 
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(len(CATEGORIES), activation = 'softmax'))


#Compiles the kernal to create the CNN
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
history = model.fit(X_train, y_train,validation_data= (X_test, y_test), epochs= 9, batch_size=20) 

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Push in one image to predict

predict_image_path = r"c:\Users\Keith Young\Desktop\Eyes_dataset\cat_eyes\CatEyes-6.jpg"
predict_image = cv2.imread(predict_image_path, cv2.IMREAD_GRAYSCALE)
predict_image = cv2.resize(predict_image, (IMG_SIZE, IMG_SIZE))
predict_image = np.array(predict_image).reshape(-1,IMG_SIZE,IMG_SIZE,1)
predict_image = predict_image / 255.0


prediction = model.predict(predict_image, batch_size=1)
predicted_class = np.argmax(prediction, axis = 1)
predicted_labels = CATEGORIES[predicted_class[0]]

print('Prediction: ', predicted_labels)

#Plotting accuracy and loss function
#Source 
# https://stackoverflow.com/questions/66785014/how-to-plot-the-accuracy-and-and-loss-from-this-keras-cnn-model

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



