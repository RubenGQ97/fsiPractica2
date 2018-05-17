import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#@author Ruben Garcia Quintana
# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f,encoding='latin1')
f.close()

train_x, train_y = train_set
validx, validy = valid_set
test_x,test_y= test_set


dataTrain_y = one_hot(train_y, 10)
dataValid_y = one_hot(validy,10)
dataTest_y = one_hot(test_y,10)


train_y= one_hot(train_y,10)    #hay diez tipos

x= tf.placeholder("float",[None,28*28])
y_ = tf.placeholder("float", [None, 10], name ='etiquetas')  # labels

W1=tf.Variable(np.float32(np.random.rand(784,10))*0.1)
b1= tf.Variable(np.float32(np.random.rand(10))*0.1)

W2=tf.Variable(np.float32(np.random.rand(10,10))*0.1)
b2=tf.Variable(np.float32(np.random.rand(10))*0.1)

h=tf.nn.sigmoid(tf.matmul(x,W1)+b1)

y=tf.nn.softmax(tf.matmul(h,W2)+b2)

loss=tf.reduce_sum(tf.square(y_ - y))

train= tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init=tf.initialize_all_variables()

sess= tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")


batch_size=20
estabilidad=0
epoch=0
error1=0;
Evalidacion=[]
Eentrenamiento=[]

while(estabilidad < 15):
    for jj in range(int(len(train_x)/batch_size)):
        batch_xs= train_x[jj * batch_size:jj * batch_size + batch_size]
        batch_ys= train_y[jj * batch_size:jj * batch_size + batch_size]
        sess.run(train,feed_dict={x: batch_xs, y_: batch_ys})

    Etrain = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size
    error2 = sess.run(loss, feed_dict={x: validx, y_: dataValid_y}) / len(dataValid_y)


    if (error2 >= error1 * 0.95):
        estabilidad += 1
    else:
        estabilidad = 0

    print("Epoch #:", epoch, "Error: ", error2, "Error Anterior: ", error1, "Estabilidad: ", estabilidad)
    Evalidacion.append(error2)
    Eentrenamiento.append(Etrain)
    error1 = error2
    epoch += 1


result = sess.run(y, feed_dict={x: test_x})
Acierta=0
falla=0

for b, r in zip(dataTest_y, result):
    if (np.argmax(b) == np.argmax(r)):
        Acierta += 1
    else:
        falla += 1

    print (b, "-->", r)
    print ("Numero de Aciertos: ", Acierta)
    print ("Numero de Fallos: ",falla)
    Total = Acierta + falla

    print("Porcentaje de aciertos: ", (float(Acierta) / float(Total)) * 100, "%")
    print("----------------------------------------------------------------------------------")

plt.plot(Evalidacion)
plt.plot(Eentrenamiento)

plt.legend(['Error Validacion', 'Error Entrenamiento'], loc='upper right')
plt.show()


# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])


# TODO: the neural net!!
