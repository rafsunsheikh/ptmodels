from ptmodels.Classifier import PreTrainedModels
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = PreTrainedModels(NUM_CLASSES=10, BATCH_SIZE=32, EPOCHS=2, LEARNING_RATE=0.001, MOMENTUM=0.9)
df = model.fit(x_train, y_train, x_test, y_test)
print(df)