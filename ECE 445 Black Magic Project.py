
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os
from PIL import Image
import serial

s = serial.Serial("COM4", 9600, timeout = 2)
cap = cv2.VideoCapture(0)
fc = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224,224)) 
    #frame = cv2.resize(frame, (480,320)) 
    height, width, channels = frame.shape 
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_skin = np.array([0, 60, 90])
    upper_skin = np.array([255, 240, 225])
    #lower_skin = np.array([0, 48, 80])
    #upper_skin = np.array([20, 255, 255])
    #15 - 45, 15 - 30, 50 - 90
    #cv2.resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC)
    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    b = cv2.blur(mask,(5,5))
    retval2,threshold = cv2.threshold(b,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Display the resulting frame
    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #new_img = np.stack((test,)*3, axis=-1)
    cv2.imshow('res',res)
    test_image = np.array(res)/255.0
    result =model.predict(test_image[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)
    if (predicted_class == 0):
        s.write(bytes("2", "utf-8"))
    print(label_names[predicted_class])
    #cv2.imshow('threshold',threshold)
    #name = "D:/gesture/training_set/right/%d.jpg"%fc
    #cv2.imwrite(name, res)
    fc += 1
    #name = "D:/gesture/test_set/test.jpg"
    #print(new_img.shape)
    #cv2.imwrite(name, threshold)
    #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[1]:


get_ipython().system('pip install -q tensorflow_hub')
from IPython.display import display 
from PIL import Image
import numpy as np


# In[2]:


from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers


# In[3]:


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}


# In[4]:


def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)
  
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))


# In[ ]:


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2" #@param {type:"string"}
def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))


# In[ ]:


data_root = r'C:\Users\Kalen\.keras\datasets\gesture\training_set'
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break


# In[ ]:


features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])
features_extractor_layer.trainable = False
model = tf.keras.Sequential([
  features_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])


# In[ ]:


import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)


# In[ ]:


result = model.predict(image_batch)
result.shape


# In[ ]:


model.compile(
  optimizer=tf.train.AdamOptimizer(), 
  loss='categorical_crossentropy',
  metrics=['accuracy'])


# In[ ]:


class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    
  def on_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])


# In[ ]:


steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=1, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])


# In[ ]:


plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)


# In[ ]:


label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names


# In[ ]:


result_batch = model.predict(image_batch)

labels_batch = label_names[np.argmax(result_batch, axis=-1)]
labels_batch


# In[ ]:


plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
_ = plt.suptitle("Model predictions")


# In[ ]:


test = r'C:\Users\Kalen\.keras\datasets\gesture\training_set\Right\645.jpg'
test = Image.open(test).resize(IMAGE_SIZE)
test = np.array(test)/255.0
test.shape


# In[ ]:


label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])


# In[ ]:


result =model.predict(test[np.newaxis, ...])
result.shape
predicted_class = np.argmax(result[0], axis=-1)
label_names[predicted_class]


# In[ ]:


# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, r"C:/Users/Kalen/.keras/model/model.ckpt")
  print("Model saved in path: %s" % save_path)


# In[ ]:


#restore
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, r"C:/Users/Kalen/.keras/model/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())


# In[ ]:




