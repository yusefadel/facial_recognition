from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json
import paho.mqtt.client as mqtt
import os
import pandas as pd
import numpy as np
import  json
from tqdm.notebook import tqdm
from PIL import Image

model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )
model_json = model.to_json()
# with open("emotiondetector.json",'w') as json_file:
#     json_file.write(model_json)
# model.save("emotiondetector.h5")

json_file = open("/content/drive/MyDrive/emotion_detection/facialemotionmodel.json", "r")###################
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("/content/drive/MyDrive/emotion_detection/facialemotionmodel.h5")######################


def resize_image(input_path, output_path, target_size=(48, 48)):
    # Open the image file
    img = Image.open(input_path)

    # Resize the image
    img_resized = img.resize(target_size, Image.ANTIALIAS)

    # Save the resized image
    img_resized.save(output_path)


def extract_features(image_path):
    feature=[]
    resize_image(image_path, output_path='/content/drive/MyDrive/emotion_detection/resized.jpg')#####################
    new_path='/content/drive/MyDrive/emotion_detection/resized.jpg'
    img = load_img(new_path, grayscale=True)
    img = np.array(img)
    img_size=img.size
    feature=img
    features = img.reshape(img_size//(48*48), 48, 48, 1)
    return features


def on_message_photo(client, userdata, msg):
    global request_counter
    # Assuming the image is sent as a byte array
    image = msg.payload

    # Save the image to a file
    with open('C:/Users/FreeComp/Desktop/stem_projects/autism_mariam/joo.JPEG', 'wb') as image_file:###########################
        image_file.write(image)
    print("image is saved ")
    
    # Increment the request counter
    request_counter += 1


    
if __name__ == '__main__':
  while 1:  
        request_counter = 0
        max_requests = 1

        client = mqtt.Client()
        
        # Assign the callback function
        client.on_message = on_message_rate

        # Connect to the broker
        client.connect("192.168.1.4")  # Change this to your MQTT broker IP/hostname #########################################

        client.subscribe("heartbeat")##################################################
        # Subscribe to the topic

        # Change this to your topic

        client.loop_start()

        try:
                # Keep the script running
            while request_counter < max_requests:
                    pass

        except KeyboardInterrupt:
                # Handle KeyboardInterrupt
                print("KeyboardInterrupt detected.")

        finally:
                # Cleanup actions
                client.loop_stop()  
                client.disconnect()
                print("Disconnected from MQTT broker")
        label = ['angry','disgust','fear','happy','neutral','sad','surprise']

        image = '/content/drive/MyDrive/emotion_detection/happy2.jpg'#######################
        feature1 = extract_features(image)
        print("original image is of sad")
        pred = model.predict(feature1)
        pred_label = label[pred.argmax()]
        print("model prediction is ",pred_label)

