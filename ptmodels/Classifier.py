import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix

from keras import callbacks
from keras import optimizers
from keras.datasets import cifar10
from keras import layers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from keras import Model
from keras.preprocessing import image
import gc



class PreTrainedModels:

    def __init__(self, 
                 NUM_CLASSES=2,
                 BATCH_SIZE = 32, 
                 EPOCHS = 1, 
                 LEARNING_RATE=1e-4, 
                 MOMENTUM=0.9):
        self.NUM_CLASSES = NUM_CLASSES
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.MOMENTUM = MOMENTUM
        self.histories = []
        # Evaluation Matrices for Training Dataset
        self.accuracies_train = []
        self.precisions_train = []
        self.recalls_train = []
        self.f1_scores_train = []
        # Evaluation Matrices for Testing Dataset
        self.accuracies_test = []
        self.precisions_test = []
        self.recalls_test = []
        self.f1_scores_test = []
        self.base_models = []
        
        self.base_models_name = [
                                'VGG16', 
                                'VGG19', 
                                "ResNet50", 
                                "ResNet50V2", 
                                "ResNet101", 
                                "ResNet101V2", 
                                "ResNet152", 
                                "ResNet152V2",
                                "MobileNet", 
                                "MobileNetV2", 
                                "DenseNet121", 
                                "DenseNet169", 
                                "EfficientNetV2B1", 
                                "EfficientNetV2B2", 
                                "EfficientNetV2B3", 
                                "EfficientNetV2S", 
                                "EfficientNetV2M", 
                                "EfficientNetV2L", 
                                "ConvNeXtTiny", 
                                "ConvNeXtSmall", 
                                "ConvNeXtBase", 
                                "ConvNeXtLarge", 
                                "ConvNeXtXLarge"
                                ]

    def fit(self, x_train, y_train, x_test, y_test):
        
        height = x_train.shape[1]
        width = x_train.shape[2]
        channel = x_train.shape[3]
        # base_model_Xception = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(height, width, channel)) ->at least 71x71
        base_model_VGG16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_VGG19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ResNet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ResNet50V2 = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ResNet101 = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ResNet101V2 = tf.keras.applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ResNet152 = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ResNet152V2 = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        # base_model_InceptionV3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(height, width, channel)) ->at least 75x75
        # base_model_InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(height, width, channel)) ->at least 75x75
        base_model_MobileNet = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_MobileNetV2 = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_DenseNet121 = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_DenseNet169 = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_DenseNet201 = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_NASNetMobile = tf.keras.applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_NASNetLarge = tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB0 = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB1 = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB2 = tf.keras.applications.EfficientNetB2(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB3 = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB4 = tf.keras.applications.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB5 = tf.keras.applications.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB6 = tf.keras.applications.EfficientNetB6(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetB7 = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2B0 = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2B1 = tf.keras.applications.EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2B2 = tf.keras.applications.EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2B3 = tf.keras.applications.EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2S = tf.keras.applications.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2M = tf.keras.applications.EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_EfficientNetV2L = tf.keras.applications.EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ConvNeXtTiny = tf.keras.applications.ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ConvNeXtSmall = tf.keras.applications.ConvNeXtSmall(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ConvNeXtBase = tf.keras.applications.ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ConvNeXtLarge = tf.keras.applications.ConvNeXtLarge(weights='imagenet', include_top=False, input_shape=(height, width, channel))
        base_model_ConvNeXtXLarge = tf.keras.applications.ConvNeXtXLarge(weights='imagenet', include_top=False, input_shape=(height, width, channel))


        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        base_models = [base_model_VGG16, 
               base_model_VGG19, 
               base_model_ResNet50, 
               base_model_ResNet50V2, 
               base_model_ResNet101, 
               base_model_ResNet101V2, 
               base_model_ResNet152, 
               base_model_ResNet152V2,
               base_model_MobileNet, 
               base_model_MobileNetV2, 
               base_model_DenseNet121, 
               base_model_DenseNet169, 
               base_model_EfficientNetV2B1, 
               base_model_EfficientNetV2B2, 
               base_model_EfficientNetV2B3, 
               base_model_EfficientNetV2S, 
               base_model_EfficientNetV2M, 
               base_model_EfficientNetV2L, 
               base_model_ConvNeXtTiny, 
               base_model_ConvNeXtSmall, 
               base_model_ConvNeXtBase, 
               base_model_ConvNeXtLarge, 
               base_model_ConvNeXtXLarge
              ]
        

        gc.collect()

        # Save the base models in main class
        for base_model in base_models:
            self.base_models.append(base_model)

        # Make the already trained layers untrainable
        for base_model in base_models:
            for layer in base_model.layers:
                layer.trainable = False

        count = 0
        for base_model in base_models:
            print("Starting Training:", self.base_models_name[count])
            count += 1

            last = base_model.output
            x = GlobalAveragePooling2D()(last)
            x = BatchNormalization()(x)
            x = Dense(256, activation='relu')(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.6)(x)
            predictions = Dense(self.NUM_CLASSES, activation='softmax')(x)

            
            model = Model(base_model.input, predictions)
            model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=self.LEARNING_RATE), metrics=['accuracy'])

            callbacks = [
                keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")
            ]
            
            history = model.fit(x_train, y_train, epochs = self.EPOCHS, callbacks = callbacks, batch_size=self.BATCH_SIZE)
            gc.collect()
            self.histories.append(history)

                
            # Model Performance for train Dataset
            y_pred = model.predict(x_train)
            y_pred = np.argmax(y_pred, axis=1)
            y_train_original = np.argmax(y_train, axis=1)

            acc_train=format(accuracy_score(y_pred, y_train_original),'.3f')
            precision_train=format(precision_score(y_train_original, y_pred, average='micro'),'.3f')
            recall_train=format(recall_score(y_train_original, y_pred, average='micro'),'.3f')
            f1_train=format(f1_score(y_train_original, y_pred, average='micro'),'.3f')
            
            #Append the training evaluation values in to lists
            self.accuracies_train.append(acc_train)
            self.precisions_train.append(precision_train)
            self.recalls_train.append(recall_train)
            self.f1_scores_train.append(f1_train)

            gc.collect()
            
            # Model Performance for test Dataset
            y_test_pred = model.predict(x_test)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            y_test_original = np.argmax(y_test, axis=1)

            acc_test=format(accuracy_score(y_test_pred, y_test_original),'.3f')
            precision_test=format(precision_score(y_test_original, y_test_pred, average='micro'),'.3f')
            recall_test=format(recall_score(y_test_original, y_test_pred, average='micro'),'.3f')
            f1_test=format(f1_score(y_test_original, y_test_pred, average='micro'),'.3f')
            
            #Append the testing evaluation values in to lists
            self.accuracies_test.append(acc_test)
            self.precisions_test.append(precision_test)
            self.recalls_test.append(recall_test)
            self.f1_scores_test.append(f1_test)
            gc.collect()

            print("appended successfully")
            print("")
            gc.collect()

        
        df = pd.DataFrame(list(zip(self.base_models_name, self.accuracies_train, self.precisions_train, self.recalls_train, self.f1_scores_train, self.accuracies_test, self.precisions_test, self.recalls_test, self.f1_scores_test)),
               columns =["Models","Accuracy train", "Precision train", "Recall train", "f1_score train", "Accuracy test", "Precision test", "Recall test", "f1_score test"])
        
        df.to_csv("Predictions.csv")
        return df

    def models_name(self):
        models = []
        models = self.base_models_name
        return models

    def train_specific_model(self, 
                 NUM_CLASSES,
                 BATCH_SIZE, 
                 EPOCHS, 
                 LEARNING_RATE, 
                 MOMENTUM, MODEL_NAME, 
                 x_train, y_train,
                 x_test, y_test, SAVE_MODEL = False):
        index = None
        for i in range(len(self.base_models_name)):
            if MODEL_NAME == self.base_models_name[i]:
                index = i

        base_model = self.base_models[index]

        for layer in base_model.layers:
                layer.trainable = False

        last = base_model.output
        x = GlobalAveragePooling2D()(last)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.6)(x)
        predictions = Dense(self.NUM_CLASSES, activation='softmax')(x)

        
        model = Model(base_model.input, predictions)
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=self.LEARNING_RATE), metrics=['accuracy'])

        callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")
        ]
        
        history = model.fit(x_train, y_train, epochs = self.EPOCHS, callbacks = callbacks, batch_size=self.BATCH_SIZE)
        gc.collect()

        if SAVE_MODEL:
            # serialize model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")

        # Model Performance for train Dataset
        y_pred = model.predict(x_train)
        y_pred = np.argmax(y_pred, axis=1)
        y_train_original = np.argmax(y_train, axis=1)

        acc_train=format(accuracy_score(y_pred, y_train_original),'.3f')
        precision_train=format(precision_score(y_train_original, y_pred, average='micro'),'.3f')
        recall_train=format(recall_score(y_train_original, y_pred, average='micro'),'.3f')
        f1_train=format(f1_score(y_train_original, y_pred, average='micro'),'.3f')
            
            #Append the training evaluation values in to lists
        self.accuracies_train.append(acc_train)
        self.precisions_train.append(precision_train)
        self.recalls_train.append(recall_train)
        self.f1_scores_train.append(f1_train)

        gc.collect()
            
            # Model Performance for test Dataset
        y_test_pred = model.predict(x_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        y_test_original = np.argmax(y_test, axis=1)

        acc_test=format(accuracy_score(y_test_pred, y_test_original),'.3f')
        precision_test=format(precision_score(y_test_original, y_test_pred, average='micro'),'.3f')
        recall_test=format(recall_score(y_test_original, y_test_pred, average='micro'),'.3f')
        f1_test=format(f1_score(y_test_original, y_test_pred, average='micro'),'.3f')
            
            #Append the testing evaluation values in to lists
        self.accuracies_test.append(acc_test)
        self.precisions_test.append(precision_test)
        self.recalls_test.append(recall_test)
        self.f1_scores_test.append(f1_test)
        gc.collect()

        print("appended successfully")
        print("")
        gc.collect()

        
        df = pd.DataFrame(list(zip(MODEL_NAME, self.accuracies_train, self.precisions_train, self.recalls_train, self.f1_scores_train, self.accuracies_test, self.precisions_test, self.recalls_test, self.f1_scores_test)),
               columns =["Model","Accuracy train", "Precision train", "Recall train", "f1_score train", "Accuracy test", "Precision test", "Recall test", "f1_score test"])
        
        df.to_csv("Predictions.csv")
        return df
        
    def evaluate_saved_model(self, x_test, y_test):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate(x_test, y_test, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    def predict_image_saved_model(self, image_path, image_width, image_height):
        img = image.load_img(image_path, target_size = (image_width, image_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        loaded_model.predict(img)
        
# Get the names of the models
# model = PreTrainedModels()
# names = []
# names = model.models_name()
# print(names)

# Train the model using a single base model
# Note: Saving models requires that you have the h5py library installed. It is usually installed as a dependency with TensorFlow. You can also install it easily as follows:
# sudo pip install h5py
# model = PreTrainedModels()
# df = model.train_specific_model(NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, MOMENTUM, MODEL_NAME, x_train, y_train, x_test, y_test, SAVE_MODEL)

# Load saved model and Evaluate
# model = PreTrainedModels()
# model.evaluate_saved_model(x_test, y_test)

# Predict single image using saved model
# model = PreTrainedModels()
# model.predict_image_saved_model(self, image_path, image_width, image_height)


