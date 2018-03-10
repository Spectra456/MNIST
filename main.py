from keras.datasets import mnist # dataset
from keras.models import Model # base class for NT
from keras.layers import Input,Dense # NT layers
from keras.utils import np_utils # for encoding

batch_size = 128 #  количество обучающих образцов, обрабатываемых одновременно за одну итерацию алгоритма градиентного спуска
num_epochs = 20
hiden_size = 512 # количество нейронов в скрытых слоях

num_train = 60000 # количество примеров в MNIST
num_test = 10000 # количество тестовых образцов в MNIST

height,width,dept = 28, 28, 1 # Свойства изображение в MNIST
num_classes = 10 # Количество классов (цифр)

(X_train, y_train),(X_test, y_test) = mnist.load_data() # Получение данных из MNIST

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test,height * width)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train,num_classes)
Y_test = np_utils.to_categorical(y_test,num_classes)

inp = Input(shape=(height * width,))
hidden_1 = Dense(hiden_size,activation="relu")(inp) # Первый слой с RELU
hidden_2 = Dense(hiden_size, activation="relu")(hidden_1)
out = Dense(num_classes,activation="softmax")(hidden_2) # Выходной слой с Softmax

model = Model(input = inp, output = out) # Задаем входной и выходной слой

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X_train,Y_train,
          batch_size=batch_size,
          nb_epoch=num_epochs,
          verbose=1,
          validation_split=0.1)

model.evaluate(X_test,Y_test,verbose=1)