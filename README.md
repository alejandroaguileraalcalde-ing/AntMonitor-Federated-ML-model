# AntMonitor-Federated-ML-model

Version: 05/05/2021

Here is the Machine Learning model that uses AntMonitor Federated 
 Here is the **model** file. 

### Model:
``` python


!pip install tensorflow==1.15.5

import tensorflow as tf
tf.__version__

filtracion_location = tf.placeholder(tf.float32, name='location_input')
filtracion_email = tf.placeholder(tf.float32, name='email_input')
filtracion_imei = tf.placeholder(tf.float32, name='imei_input')
filtracion_device = tf.placeholder(tf.float32, name='device_input')
filtracion_serialnumber = tf.placeholder(tf.float32, name='serialnumber_input')
filtracion_macaddress = tf.placeholder(tf.float32, name='macaddress_input')
filtracion_advertiser = tf.placeholder(tf.float32, name='advertiser_input')

#destinos: 5 posibles--> Internal, Ads, Analytics, Sns, Develop             
destino_internal = tf.placeholder(tf.float32, name='internal_dst_input')
destino_ads = tf.placeholder(tf.float32, name='ads_dst_input')
destino_analytics = tf.placeholder(tf.float32, name='analytics_dst_input')
destino_sns = tf.placeholder(tf.float32, name='sns_dst_input')
destino_develop = tf.placeholder(tf.float32, name='develop_dst_input')


y_ = tf.placeholder(tf.float32, name='target')

#distintos pesos para cada dato: nombre= Px
Plocation = tf.Variable(90., name='Plocation') 
Pemail = tf.Variable(80., name='Pemail') 
Pimei = tf.Variable(30., name='Pimei') 
Pdevice = tf.Variable(20., name='Pdevice') 

Pserialnumber = tf.Variable(30., name='Pserialnumber') 
Pmacaddress = tf.Variable(10., name='Pmacaddress') 
Padvertiser = tf.Variable(50., name='Padvertiser') 

#Pesos para cada destino: 
Pinternal_dst = tf.Variable(1., name='Pinternal_dst') 
Pads_dst = tf.Variable(8., name='Pads_dst') 
Panalytics_dst = tf.Variable(3., name='Panalytics_dst') 
Psns_dst = tf.Variable(9., name='Psns_dst') 
Pdevelop_dst = tf.Variable(1., name='Pdevelop_dst') 



y_aux1 = tf.add(tf.multiply(filtracion_location, Plocation), 0.0)
y_aux2 = tf.add(tf.multiply(filtracion_email, Pemail), y_aux1)
y_aux3 = tf.add(tf.multiply(filtracion_imei, Pimei), y_aux2)
y_aux4 = tf.add(tf.multiply(filtracion_device, Pdevice), y_aux3)
y_aux13 = tf.add(tf.multiply(filtracion_serialnumber, Pserialnumber), y_aux4)
y_aux14 = tf.add(tf.multiply(filtracion_macaddress, Pmacaddress), y_aux13)


#destinos
y_aux1_dst = tf.add(tf.multiply(destino_internal, Pinternal_dst), 0.0)
y_aux2_dst = tf.add(tf.multiply(destino_ads, Pads_dst), y_aux1_dst)
y_aux3_dst = tf.add(tf.multiply(destino_analytics, Panalytics_dst), y_aux2_dst)
y_aux4_dst = tf.add(tf.multiply(destino_sns, Psns_dst), y_aux3_dst)
y_aux5_dst = tf.add(tf.multiply(destino_develop, Pdevelop_dst), y_aux4_dst)


y_aux_final1 = tf.add(tf.multiply(filtracion_advertiser, Padvertiser), y_aux14)

y = tf.multiply(y_aux_final1, y_aux5_dst)

#salida del modelo
y = tf.identity(y, name='output')



#loss           

loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()

# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.

saver_def = tf.train.Saver().as_saver_def()


with open('graph_Ant_v16.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())


print('Operation to initialize variables:       ', init.name)
print('Tensor to feed as input data:            ', filtracion_location.name)
print('Tensor to feed as input data:            ', filtracion_email.name)
print('Tensor to feed as input data:            ', filtracion_imei.name)
print('Tensor to feed as input data:            ', filtracion_device.name)
"""
print('Tensor to feed as input data:            ', umbral.name)

print('Tensor to feed as input data:            ', Plocation.name)
print('Tensor to feed as input data:            ', Pemail.name)
print('Tensor to feed as input data:            ', Pimei.name)
print('Tensor to feed as input data:            ', Pdevice.name)


#print('Tensor to feed as input data:            ', Pdni.name)
#print('Tensor to feed as input data:            ', Pphone.name)
print('Tensor to feed as input data:            ', Pserialnumber.name)
print('Tensor to feed as input data:            ', Pmacaddress.name)
print('Tensor to feed as input data:            ', Padvertiser.name)
"""
print('Tensor to feed as training targets:      ', y_.name)
print('Tensor to fetch as prediction:           ', y.name)
print('Operation to train one step:             ', train_op.name)
print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
print('Tensor to read value of Peso location                ', Plocation.value().name)
print('Tensor to read value of Peso email                ', Pemail.value().name)
print('Tensor to read value of Peso imei                ', Pimei.value().name)
print('Tensor to read value of Peso device                ', Pdevice.value().name)
#print('Tensor to read value of umbral decisión                ', umbral.value().name)
#print('Tensor to read value of umbral decisión                ', aux.value().name)
print('Trainable variables: ', tf.trainable_variables())
print('Loss:       ', loss.name)

print('myvar.initializer:          ', Plocation.initializer.name)
print('myvar.initializer.inputs[1]:', Plocation.initializer.inputs[1].name)

print('myvar.initializer:          ', Pemail.initializer.name)
print('myvar.initializer.inputs[1]:', Pemail.initializer.inputs[1].name)


saver = tf.train.Saver()
#Training
#saver.save(sess, your_path + "/checkpoint_name.ckpt")
#TensorFlow session
sess = tf.Session()
sess.run(init)
saver.save(sess, "/content/sample_data/h"+"/checkpoint_name_Ant_v16.ckpt")
```

### Important information: 

In order to use this model in an Android app, you can **only use TF libraries, not keras or others**.

### References: 

 (1)[TensorFlow](https://www.tensorflow.org/)

 

