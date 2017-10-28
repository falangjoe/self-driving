import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
    
def load_log_file(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        result = [[line[0],line[1],line[2],float(line[3])] for line in reader]
        return result
    
def load_log_files():
    folders = [6]
    result = []
    
    for ii in folders:
        file = './data/{0}/driving_log.csv'.format(ii)
        log = load_log_file(file) 
        result += log
    return result
    
def flip_frame(frame):
    image_flipped = np.fliplr(frame[0])
    angle_flipped = -frame[1]
    return [image_flipped,angle_flipped]


def load_frames(log):
    
    offset = 0.1 #* (1.0 / 25.0)
    
    center_angle = log[3]
    center_image = cv2.imread(log[0])
    center_frame = [center_image,center_angle]
    yield center_frame
    yield flip_frame(center_frame)
 
    
    #left_image = cv2.imread(log[1])
    #left_angle = center_angle - offset
    #left_frame = [left_image,left_angle] 
    #yield left_frame
##
    #right_image = cv2.imread(log[2])
    #right_angle = center_angle + offset
    #right_frame = [right_image,right_angle]
    #yield right_frame
  
    
    
train_logs,valid_logs = train_test_split(load_log_files(),test_size=0.1)

def generate_frames(logs):
    logs = shuffle(logs)
    for log in logs:
        frames = load_frames(log)
        for frame in frames:
            yield frame
            
            
batch_size = 128
            
def generator(logs):
    
    while True:
        frames = generate_frames(logs)
        
        inputs = []
        targets = []
        
        for frame in frames:
            inputs.append(frame[0])
            targets.append(frame[1])
            if(len(targets) >= batch_size):
                yield np.array(inputs), np.array(targets)
                inputs = []
                targets = []
                

            
                
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(1 * 18, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(1 * 18, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(1 * 18, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1 * 360,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1 * 120,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1 * 10,activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    generator(train_logs),
    2 * len(train_logs) // batch_size,
    epochs=5,
    validation_data = generator(valid_logs),
    validation_steps = 2 * len(valid_logs) // batch_size)

#model.fit(X_train,y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

    

        

