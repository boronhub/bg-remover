import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from PIL import Image
import numpy as np
import cv2
from camvid.mapping import decode
from tiramisu.model import create_tiramisu

input_shape = (224, 224, 3)
number_classes = 32  # CamVid data consist of 32 classes

img_input = Input(shape=input_shape)
x = create_tiramisu(number_classes, img_input)
model = Model(img_input, x)
model.load_weights('models/my_tiramisu.h5')

def predict(image):
    prediction = model.predict(image[None, :, :, :])
    prediction = prediction.reshape((224,224, -1))
    return prediction

def main():
    print()
    print('##################################################################')
    print()
    path = input('Enter path of file: ')
    print()
    print('##################################################################')
    print()
    print("Removing background...")
    print()
    image = Image.open(path)
    prediction = predict(image[:, :, 0:3])
    print('##################################################################')

    prediction = image1.resize(prediction[:, :, 1], (image.height, image.width))
    prediction[prediction>0.5*255] = 255
    prediction[prediction<0.5*255] = 0

    transparency = np.append(np.array(image)[:, :, 0:3], prediction[: , :, None], axis=-1)
    png = Image.fromarray(transparency)

    png.save('output.png')

    print()
    print("saved the output image in output.png")
    print()
    print('##################################################################')

if __name__ == '__main__':
    main()
