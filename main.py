import flask
import os
from flask import Flask, redirect 
from flask import render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename("image.jpg")))
      return render_template('index2.html')

@app.route('/code')
def code():
     base = request.args.get('choice1')
     attack = request.args.get('choice2')
     defense = request.args.get('choice3')

     if (base == 'MNIST'): 
          import tensorflow as tf
          from PIL import Image
          import cv2
          tf.compat.v1.disable_eager_execution()
          from keras.models import Sequential
          from keras.layers import Dense, MaxPooling2D
          from keras.datasets import mnist
          from keras.utils import to_categorical
          import numpy as np
          from keras.layers import Conv2D, Flatten
          import matplotlib.pyplot as plt
          import random

          (X_train, y_train), (X_test, y_test)=mnist.load_data()
          #Normalisation et redimension des valeurs
          X_train=X_train/255.
          X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
          X_test=X_test/255.
          X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
          #One-hot-encoder
          y_train_ohe=to_categorical(y_train)
          y_test_ohe=to_categorical(y_test)
          X_test = X_test[0]
          from art.attacks.evasion import FastGradientMethod
          from art.estimators.classification import KerasClassifier
          import os
          import pandas as pd
          import tensorflow as tf
          import keras
          from keras.preprocessing import image
          #from kaggle_datasets import KaggleDatasets
          import matplotlib.pyplot as plt
          import PIL
          cnn_model = keras.models.load_model("./cnn_model_trained.h5")
          classifier = KerasClassifier(model= cnn_model ,clip_values=(0,1),use_logits=False)
          width = 28
          height = 28
          dim = (width, height)

          file = "./static/image.jpg"
          image = cv2.imread(file, 0)
          image = cv2.resize(image, (28, 28))
          image = image.astype('float32')
          height, width = image.shape
          print (height, width)
          image = image.reshape(1, 28, 28, 1)
          image = 255-image
          image /= 255
          preds = np.argmax(classifier.predict(image), axis=1)
          print(preds)
          #return "finito"
          if (defense == 'NONE' and attack == 'BIM'):
               from art.attacks.evasion import BasicIterativeMethod
               bim_crafter = BasicIterativeMethod(estimator=classifier, eps=2, max_iter=3)
               X_test_bim = bim_crafter.generate(x=image)
               bim_preds = np.argmax(classifier.predict(X_test_bim), axis=1)
               image_test_save = cv2.resize(X_test_bim[0], (200, 200))
               cv2.imwrite('./static/modified_image.jpg', image_test_save*255)
               return redirect("/post_process?attack="+str(attack)+"&defense="+str(defense)+"&result="+str(preds)+"&result1="+str(bim_preds)+"&result2=NULL")
          
          elif (defense == 'NONE' and attack == 'PATCH'):
               from art.attacks.evasion import  AdversarialPatch
               patch_crafter = AdversarialPatch(classifier, targeted = False, max_iter = 3)
               X_test_patch = patch_crafter.apply_patch(image, 0.2)
               patch_preds = np.argmax(classifier.predict(X_test_patch), axis=1)
               image_test_save = cv2.resize(X_test_patch[0], (200, 200))
               cv2.imwrite('./static/modified_image.jpg', image_test_save*255)
               return redirect("/post_process?attack="+str(attack)+"&defense="+str(defense)+"&result="+str(preds)+"&result1="+str(patch_preds)+"&result2=NULL")

          elif (defense == 'NONE' and attack == 'FGSM'):
               adv_crafter = FastGradientMethod(classifier, eps=2)
               X_test_fgsm = adv_crafter.generate(x=image)
               preds_fgsm = np.argmax(classifier.predict(X_test_fgsm), axis=1)
               print(preds_fgsm)
               image_test_save = cv2.resize(X_test_fgsm[0], (200, 200))
               cv2.imwrite('./static/modified_image.jpg', image_test_save*255)
               return redirect("/post_process?attack="+str(attack)+"&defense="+str(defense)+"&result="+str(preds)+"&result1="+str(preds_fgsm)+"&result2=NULL")
          

          elif (defense == 'AdversarialTrainer' and attack == 'BIM'):
               from art.defences.trainer import AdversarialTrainer
               from art.attacks.evasion import  AdversarialPatch

               adv_crafter = FastGradientMethod(classifier, eps=2)
               bim_crafter = BasicIterativeMethod(estimator=classifier, eps=2, max_iter=3)
               X_test_bim = bim_crafter.generate(x=image)
               robust_classifier = AdversarialTrainer(classifier,[adv_crafter, bim_crafter],ratio=0.5)
               robust_classifier.fit(X_train, y_train_ohe)
               from art.attacks.evasion import BasicIterativeMethod
               robust_preds_bim = np.argmax(robust_classifier.predict(X_test_bim), axis=1)
               print(robust_preds_bim)

          elif (defense == 'AdversarialTrainer' and attack == 'PATCH'):
               from art.defences.trainer import AdversarialTrainer
               from art.attacks.evasion import  AdversarialPatch
               from art.attacks.evasion import BasicIterativeMethod


               adv_crafter = FastGradientMethod(classifier, eps=2)
               patch_crafter = AdversarialPatch(classifier, targeted = False, max_iter = 3)
               bim_crafter = BasicIterativeMethod(estimator=classifier, eps=2, max_iter=3)

               X_test_patch = patch_crafter.apply_patch(image, 0.2)
               robust_classifier = AdversarialTrainer(classifier,[adv_crafter, patch_crafter],ratio=0.5)
               robust_classifier.fit(X_train, y_train_ohe)
               robust_preds_fgsm = np.argmax(robust_classifier.predict(image), axis=1)
               print(robust_preds_fgsm)

          elif (defense == 'AdversarialTrainer' and attack == 'FGSM'):
               from art.defences.trainer import AdversarialTrainer
               from art.attacks.evasion import BasicIterativeMethod
               adv_crafter = FastGradientMethod(classifier, eps=2)
               bim_crafter = BasicIterativeMethod(estimator=classifier, eps=2, max_iter=3)
               X_test_fgsm = adv_crafter.generate(x=image)
               robust_classifier = AdversarialTrainer(classifier,[adv_crafter, bim_crafter],ratio=0.5)
               robust_classifier.fit(X_train, y_train_ohe)
               robust_preds_fgsm = np.argmax(robust_classifier.predict(X_test_fgsm), axis=1)
               print(robust_preds_fgsm)


          elif (defense == 'Gaussian Augmentation' and attack == 'BIM'):
               from art.defences.preprocessor import GaussianAugmentation
               from art.attacks.evasion import BasicIterativeMethod

               ga = GaussianAugmentation(ratio=0.5, clip_values=(0, 1))
               bim_crafter = BasicIterativeMethod(estimator=classifier, eps=2, max_iter=3)
               gaussian_model = KerasClassifier(model= cnn_model ,preprocessing_defences = [ga])
               gaussian_model.fit(X_train, y_train_ohe, nb_epochs=5)
               X_test_bim = bim_crafter.generate(x=image)
               gaussian_preds = np.argmax(gaussian_model.predict(X_test_bim), axis=1)
               print(gaussian_preds)
               image_test_save = cv2.resize(X_test_bim[0], (200, 200))
               cv2.imwrite('modified_image_def_bim.jpg', image_test_save*255)
               return 7
          elif (defense == 'Gaussian Augmentation' and attack == 'PATCH'):
               from art.defences.preprocessor import GaussianAugmentation
               from art.attacks.evasion import  AdversarialPatch
               ga = GaussianAugmentation(ratio=0.5, clip_values=(0, 1))
               gaussian_model = KerasClassifier(model= cnn_model ,preprocessing_defences = [ga])
               gaussian_model.fit(X_train, y_train_ohe, nb_epochs=5)
               from art.attacks.evasion import  AdversarialPatch
               patch_crafter = AdversarialPatch(classifier, targeted = False, max_iter = 3)
               X_test_patch = patch_crafter.apply_patch(image, 0.2)
               patch_preds = np.argmax(gaussian_model.predict(X_test_patch), axis=1)
               image_test_save = cv2.resize(X_test_patch[0], (200, 200))
               cv2.imwrite('modified_image_patch.jpg', image_test_save*255)
               return
          elif (defense == 'Gaussian Augmentation' and attack == 'FGSM'):
               from art.defences.preprocessor import GaussianAugmentation
               ga = GaussianAugmentation(ratio=0.5, clip_values=(0, 1))
               gaussian_model = KerasClassifier(model= cnn_model ,preprocessing_defences = [ga])
               gaussian_model.fit(X_train, y_train_ohe, nb_epochs=5)
               adv_crafter = FastGradientMethod(classifier, eps=2)
               X_test_fgsm = adv_crafter.generate(x=image)
               preds_fgsm = np.argmax(gaussian_model.predict(X_test_fgsm), axis=1)
               image_test_save = cv2.resize(X_test_fgsm[0], (200, 200))
               cv2.imwrite('modified_image.jpg', image_test_save*255)
               return "/"
     elif(base == 'ALZHEIMER'):
          return render_template('post_process_att_f.html')
     elif (base == 'ALZHEIMER' & attack == 'PATCH'):
          return render_template('post_process_att_p.html')
     elif (base == 'ALZHEIMER' & attack == 'BIM'):
          return render_template('post_process_a_b.html')
     
@app.route('/post_process')
def post_process():
          return render_template("post_process.html")


