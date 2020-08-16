from keras.models import Model, load_model
import cv2
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

model1_path = "epoch_3dcnn_84.hdf5"
classifier = load_model('epoch_3dcnn_84.hdf5')
img_row, img_height, img_depth = 32, 32, 3

img = image.load_img('chilli.jpg', target_size=(32, 32))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)
classes = ["Broccoli", "Cabbage", "Capsicums", "Carrots", "Cauliflower", "Celeriac", "Celery", "Chilli peppers",
           "Chokos", "Courgettes and Scallopini", "Cucumber", "Eggplant", "Fennel", "Fresh  garnishes and flowers",
           "Garlic", "Ginger", "Indian vegetables", "Kale and Cavolo Nero", "Kohlrabi", "Melons", "Mushrooms", "Okra",
           "Onions", "Potatoes", "Pumpkins", "Radishes", "Spinach", "Spring onions", "Sweet corn", "Tomatoes",
           "Turnips", "Yams"]
classes = {i: classes[i] for i in range(0, len(classes))}
# print(classes)
output = classifier.predict(img)
# print(output[0])
out = []
for i in output[0]:
    out.append(i)
# print(out)
teet = {}
for i in range(len(out)):
    teet[classes[i]] = out[i]

final = sorted(teet.items(), key=lambda x: x[1], reverse=True)
final = dict(final)

first3pairs = {k: final[k] for k in list(final)[:3]}

plt.bar(range(len(first3pairs)), list(first3pairs.values()), align='center')
plt.xticks(range(len(first3pairs)), list(first3pairs.keys()))
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x
plt.title('AlexNet')
plt.show()

# output = np.argmax(output,axis = 1)
# print(classes[int(output[0])])
