
import  numpy as np

from keras.preprocessing import image
from keras.models import load_model


image_shape = (224, 224, 3)

train_path='C://Users/Asus/Desktop/Plant-Disease-New/Plant-Disease/train'
test_path='C://Users/Asus/Desktop/Plant-Disease-New/Plant-Disease/test'
batch_size=32


M=load_model('Plant_TLKN.h5')
my_image = image.load_img('80bdc2a3-66b8-4d95-abe7-65c7806f7cbf___RS_LB 3073.JPG', target_size=image_shape)
my_image_array = image.img_to_array(my_image)
my_image_array = np.expand_dims(my_image_array, axis=0)
pred = (M.predict(my_image_array))

print(pred)
print(np.argmax(pred))




class_disease=['pepper_bacteria', 'pepper_healthy', 'potato_earlyblight', 'potato_healthy', 'potato_lateblight', 'tomato_bacterialspot', 'tomato_earlyblight', 'tomato_healthy', 'tomato_targetspot']

print(class_disease[np.argmax(pred)])
