from sklearn import svm
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import os 


X_train =  []
y_train = []

for filename in os.listdir("./data/images"):

   if filename == '.DS_Store':
      continue
   im  = skio.imread(os.path.join("./data/images", filename), plugin="pil")
   image_resized = resize(im, (2150, 2205), anti_aliasing=True)
   if image_resized is not None:
      print(image_resized.shape)
      X_train.append(image_resized)

X_train = np.array(X_train)
n, x, y = X_train.shape
print(n, x, y)
X_train = np.reshape(X_train, (n, x*y))
print(X_train.shape)
for x in range(0, 9):
   y_train.append(1)

for x in range(0, 11):
   y_train.append(0)

y_train = np.array(y_train)

"""
for im in X_train:
   plt.imshow(im)
   plt.show()

"""
clf = svm.SVC(random_state=0, kernel = "rbf")
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))