import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import code




TRAIN_DIR = '/Users/michaelgaras/dev/venv/Celloscope/Celloscope-tensorflow/train' 
TEST_DIR = '/Users/michaelgaras/dev/venv/Celloscope/Celloscope-tensorflow/test'
IMG_SIZE = 50
LR = 1e-3




MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match

#print("Model name is ",MODEL_NAME)
#print("os.listdir:",tqdm(os.listdir(TEST_DIR)))


def label_img(img):
    word_label = img.split('.')[-3]  # label the image by either 'cat' or 'dog'
    print("Word label is ", word_label)
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]



def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)): #loop through all photos in the training set directory
        label = label_img(img) #label the picture
        path = os.path.join(TRAIN_DIR,img) #grab full image path
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #convert to to greyscale
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #convert the size to IMG SIZE x IMG SIZE
        training_data.append([np.array(img),np.array(label)]) #create the array of images with the labelled tag
    shuffle(training_data)
    np.save('train_data.npy', training_data) #save the numpy array to be accessed later
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

#train_data = create_train_data()   #only use this is if it is a new data set
# If you have already created the dataset:
train_data = np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import tensorflow as tf
tf.reset_default_graph()



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input') #input size is IMG SIZE x IMG SIZE

#CREATING CONVOLUTION NETWORKS!! THIS IS THE KEY TO SUCCESS

convnet = conv_2d(convnet, 32, 5, activation='relu')  #first convolution network layer
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu') #second convolution network layer
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = fully_connected(convnet, 1024, activation='relu') #fully connected convolution layer
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax') #this is the OUTPUT. 2 here stands for 2 possible outcomes, dog or cat.
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log') #create the model



if os.path.exists('{}.meta'.format(MODEL_NAME)): #check if model exists already
     model.load(MODEL_NAME)
     print('\033[93m' + 'MODEL IS LOADED AND READY TO PREDICT!' + '\033[0m')


train = train_data[:-500] #training data used will be all pictures in training data set except the last 500
test = train_data[-500:] #test data used will be the first 500 pictures in the training data set


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1) #train has both the image data and the image label so i[0] is the image data
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


#THIS IS WHERE WE TRAIN THE MODEL!!! -- CRITICAL
#model.fit({'input': X}, {'targets': Y}, n_epoch=6, validation_set=({'input': test_x}, {'targets': test_y}),
#    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#You can view the model graphs by running this command in your terminal:
# $ tensorboard --logdir=/Users/michaelgaras/dev/venv/Celloscope/Celloscope-tensorflow/log
# then navigating to localhost:6006


#model.save(MODEL_NAME)


import matplotlib.pyplot as plt

# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:
test_data = np.load('test_data.npy')

fig=plt.figure()
for num,data in enumerate(test_data[0:12]): #iterate through the first 12 pictures and plot them
    # cat: [1,0]
    # dog: [0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1) 
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]   #This is where the model actually predicts if it's a cat or a dog. model_out will return an array of two elements, the first element is how likely is it a dog eg: [ 0.25, 0.75]
    print("Model prediction is: ",model_out)
    print("Model argmax is: ",np.argmax(model_out))
    #THIS IS PYTHON EQUIVALENT OF BINDING.PRY -- VERY USEFUL
    # code.interact(local=locals())
    if np.argmax(model_out) == 1: str_label="DOG "
    else: str_label="CAT"

    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


# with open('submission_file.csv','w') as f:
#     f.write('id,label\n')

# with open('submission_file.csv','a') as f:  # create a submission excel sheet -- not required for our Celloscope purpose
#     for data in tqdm(test_data):
#         img_num = data[1]
#         img_data = data[0]
#         orig = img_data
#         data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#         model_out = model.predict([data])[0]
#         f.write('{},{}\n'.format(img_num,model_out[1]))
