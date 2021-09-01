####################################################################################
#
# Author : Pakshal Shashikant Jain
# Date : 29/06/2021
# program : Machine Learning of Fashion_Mnist Dataset using Tensorflow and Keras 
#
####################################################################################

# Required Python Packages 

import tensorflow as tf 

import numpy as np
import matplotlib.pyplot as plt 

######################################################################################
def Display(train_images,test_images,class_names,train_labels) :
    """
    Display Images of Training and Testing Data set 
    :param : train_images
    :param : test_images 
    :param : class_names 
    :param : train_labels 
    :return : None 
    """
    # Display one graphical image from data using Matplot Lib

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Conversion of Image from Colored Images To Gray Scale Image 
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Dispaly First 25 Data in image format (after Converting Color Image in Gray Scale Format)

    plt.figure(figsize = (10,10))
    for i in range(25) :
        plt.subplot(5,5,i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i],cmap = plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()  


def plot_image(i,predictions_array,true_label,img,class_names) :
    """
    Display Predictions(actual Training and Testing Output) in Graphical Format 
    :param : i 
    :param : predictions_array
    :param : true_label
    :param : img 
    :param : class_names 
    :return : None 
    """
    true_label = true_label[i]
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img,cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'green'
    else :
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),(class_names[true_label]),color = color))

def plot_value_array(i,predictions_array,true_label) :
    """
    Display Predictions(ouput) is Correct or Worng Using Bar graph present in matplotlib
    :param : i
    :param : prediction_array 
    :param : true_label
    """
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_array,color = "#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

#################################################################################################################

def main() :
    print("Ganapati Bappa Moraya.........")
    print(tf.__version__)

    # Load The Data
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

    #Class Names To Plot The Data
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

    # Details of Data 
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    # Print First Five Records of Labels 
    print(train_labels)
    print(test_labels)

    # Display Images of Training Dataset 
    Display(train_images,test_images,class_names,train_labels)

    # Setup the layers 

    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)),tf.keras.layers.Dense(128, activation = 'relu'),tf.keras.layers.Dense(10)])

    model.compile(optimizer = 'adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics = ['accuracy'])

    model.fit(train_images,train_labels,epochs = 10)

    test_loss,test_acc = model.evaluate(test_images,test_labels,verbose = 1)

    print("Test Accuracy : ",test_acc)
    
    # Make Predicitons 

    probablity_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])


    predictions = probablity_model.predict(test_images)

    print(predictions[0])

    pred = np.argmax(predictions[0])
    print("Maximum time prediction is of :",pred)

    # Verify Predictions

    i = 0
    plt.figure(figsize = (6,3))
    plt.subplot(1,2,1)
    plot_image(i,predictions[i],test_labels,test_images,class_names)
    plt.subplot(1,2,2)
    plot_value_array(i,predictions[i],test_labels)
    plt.show()

    num_rows = 5
    num_cols = 3

    num_images = num_rows * num_cols
    plt.figure(figsize = (2*2*num_cols,2 * num_rows))
    for i in range(num_images) :
        plt.subplot(num_rows,2 * num_cols, 2 * i + 1)
        plot_image(i,predictions[i],test_labels,test_images,class_names)
        plt.subplot(num_rows,2 * num_cols,2 * i + 2)
        plot_value_array(i,predictions[i],test_labels)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__" :
    main()