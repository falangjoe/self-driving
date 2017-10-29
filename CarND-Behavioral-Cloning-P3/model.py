import csv
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_frames(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            position = float(line[3])
            angle = 0.25 * (abs(position) // 0.01)
            middle,left,right = line[0],line[1],line[2]
            yield (angle, position, middle)
            yield (angle, position + 0.25, left)
            yield (angle, position - 0.25, right)
            #yield (angle, position, middle, left, right)

def group_frames(frames):
    angles = list(map(lambda v : v[0], frames))
    keys = list(set(angles))
    samples = dict((k, []) for k in keys)
    for frame in frames:
        samples[frame[0]].append(frame)
    return samples

def pick_frames(samples):
    
    sample_count = 600

    picked_frames = []
    for key in samples.keys():
        frames = samples[key]
        indicies = np.random.permutation(np.arange(len(frames)))[0:sample_count]
        for ii in indicies:
            picked_frames.append(frames[ii])
            
    return picked_frames

def flip_frame(frame):
    image_flipped = np.fliplr(frame[0])
    angle_flipped = -frame[1]
    return [image_flipped,angle_flipped]

def load_frame(frame):
    center_position = frame[1]
    center_image = cv2.imread(frame[2])
    center_frame = [center_image,center_position]
    if random.choice([True, False]):
        return center_frame
    else:
        return flip_frame(center_frame)
    
def pick_training_data():
    all_frames = list(read_frames('./data/6/driving_log.csv'))
    samples =  group_frames(all_frames) 
    frames = pick_frames(samples)
    return frames
    

train_logs,valid_logs = train_test_split(pick_training_data(),test_size=0.1)

def generate_frames(logs):
    logs = shuffle(logs)
    for log in logs:
        frame = load_frame(log)
        yield frame
            
batch_size = 128
            
def generator(logs):
    
    while True:
        frames = generate_frames(logs)
        
        inputs = []
        targets = []
        
        for frame in frames:
            #print('------------')
            #print(frame)
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

#model = Sequential()
#model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Conv2D(1 * 6, (5, 5), activation="relu"))
#model.add(MaxPooling2D())
#model.add(Conv2D(1 * 16, (5, 5), activation="relu"))
#model.add(MaxPooling2D())
##model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(1 * 120,activation="relu"))
##model.add(Dropout(0.5))
#model.add(Dense(1 * 84,activation="relu"))
##model.add(Dropout(0.5))
#model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    generator(train_logs),
    len(train_logs) // batch_size,
    epochs=10,
    validation_data = generator(valid_logs),
    validation_steps = len(valid_logs) // batch_size)

#model.fit(X_train,y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

    

        

