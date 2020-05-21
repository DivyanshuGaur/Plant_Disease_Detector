
import  numpy as np

from keras.preprocessing import image

from keras.layers import Dense,Flatten
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from matplotlib.image import imread
import os


train_path='C://Users/Asus/Desktop/Plant-Disease-New/Plant-Disease/train'
test_path='C://Users/Asus/Desktop/Plant-Disease-New/Plant-Disease/test'
batch_size=32

def classify():


    li=os.listdir(train_path+'/potato_earlyblight')
    img_array=imread(train_path+'/potato_earlyblight/'+li[0])
    plt.imshow(img_array)
    #plt.show()

    print(img_array.shape)
    image_shape=(224,224,3)



    train_gen=ImageDataGenerator( rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True)



    test_gen=ImageDataGenerator(rescale= 1./255)


    train_datagen=train_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                batch_size=32,
                                                color_mode='rgb',
                                                class_mode='categorical'
                                                )


    test_datagen=train_gen.flow_from_directory(test_path,
                                                target_size=image_shape[:2],
                                                batch_size=32,
                                                color_mode='rgb',
                                                class_mode='categorical'
                                                )

    train_num = train_datagen.samples
    valid_num = test_datagen.samples

    vgg16=VGG16(input_shape=(224,224,3),include_top=False)

    model=Sequential()
    for layer in vgg16.layers:
        model.add(layer)

    for layer in model.layers:
        layer.trainable=False

    model.add(Flatten())

    model.add(Dense(9,activation='softmax'))


    #print(model.summary())

  #  model.compile(optimizer='adam',
   #                    loss='categorical_crossentropy',
    #                   metrics=['accuracy'])

    #history = model.fit(train_datagen,
     #                        steps_per_epoch=train_num // batch_size,
      #                       validation_data=test_datagen,
       #                      epochs=5,
        #                     validation_steps=valid_num // batch_size,
         #                    )



    #model.save('Plant_TL.h5')

    my_image = image.load_img('83', target_size=image_shape)
    my_image_array = image.img_to_array(my_image)
    my_image_array = np.expand_dims(my_image_array, axis=0)
    pred = (model.predict_classes(my_image_array))


    print(pred)











    pass



if __name__ == '__main__':
    classify()