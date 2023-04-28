from tensorflow.keras.layers import MaxPooling2D, Conv2D
import numpy as np
import tensorflow as tf

# Step 2
def net48_siso(): 
    height, width = 48, 48
    inputs = tf.keras.Input((height, width, 3));
    x = Conv2D(8,3,1,'same',activation='relu')(inputs)
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(2,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(2,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(2,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(2,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(2,3,1,'same',activation='relu')(x)
    
    x = Conv2D(2,3,1, padding='same',activation='softmax',name='cls_output')(x)
    outputs = x
    return tf.keras.Model(inputs = inputs, outputs = outputs)
# Step 3 
def net48_simo(): 
    height, width = 48, 48
    inputs = tf.keras.Input((height, width, 3));
    x = Conv2D(8,3,1,'same',activation='relu')(inputs)
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(4,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(4,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(4,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(4,3,1,'same',activation='relu')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2,padding='valid')(x)
    x = Conv2D(4,3,1,'same',activation='relu')(x)
    
    cls = Conv2D(2, 3, 1, padding='same', 
               activation='softmax', 
               name='cls_output')(x)
    reg = Conv2D(4, 3, 1, padding='same', name='reg_output')(x)
    outputs = x
    return tf.keras.Model(inputs = inputs, outputs = [cls,reg])

def check_model(cnn_model, is_simo=False):
  print('Checking the CNN model...')
  is_passed = True
  is_cls_output = False
  is_reg_output = False
  ishape = cnn_model.input_shape[1:]
  if (48,48,3) != ishape:
     print('Error! The input shape', ishape, 'is not equal to (48,48,3)')
     is_passed = False
  for i,layer in enumerate(cnn_model.layers):
     if layer.name == 'cls_output':
       oshape = layer.output_shape[1:]
       is_cls_output = True
       if (1,1,2) != oshape:
          print('Error! The output shape', oshape, 'is not equal to (1,1,2)')
          is_passed = False
     if layer.name == 'reg_output':
       oshape = layer.output_shape[1:]
       is_reg_output = True
       if (1,1,4) != oshape:
          print('Error! The output shape', oshape, 'is not equal to (1,1,4)')
          is_passed = False
     layer_type = layer.__class__.__name__
     if layer_type == 'Dense':
       print('Error! Cannot use Dense layers.')
       is_passed = False
     if layer_type == 'Conv2D':
       kernel_size = layer.kernel_size
       strides = layer.strides
       if np.max(kernel_size) > 7:
         print('({:d} {:s}) Error! kernel_size should be less than or equal to 7'.format(i, layer_type))
         is_passed = False
       if np.max(strides) > 2:
         print('({:d} {:s}) Error! stride should be less than or equal to 2'.format(i, layer_type))
         is_passed = False
     if layer_type == 'MaxPooling2D':
       pool_size = layer.pool_size
       strides = layer.strides
       if np.max(pool_size) > 2:
         print('({:d} {:s}) Error! pool_size should be less than or equal to 2'.format(i, layer_type))
         is_passed = False
       if np.max(strides) > 2:
         print('({:d} {:s}) Error! stride should be less than or equal to 2'.format(i, layer_type))
         is_passed = False
  if not is_cls_output:
     print('Error! There should be a Conv. layer with name cls_output')
     is_passed = False
  if is_simo:
     if not is_reg_output:
       print('Error! There should be a Conv. layer with name reg_output')
       is_passed = False
  if is_passed:
     print('Great! The CNN model satisfies all the requirements.')

model48_siso = net48_siso()
model48_siso.summary()
check_model(model48_siso)
print()

model48_simo = net48_simo()
model48_simo.summary()
check_model(model48_simo, is_simo=True)
print()

img = np.random.randn(1, 48, 48, 3)
outputs = model48_simo.predict(img)
print(outputs[0].shape) # should display (1,1,1,2)
print(outputs[1].shape) # should display (1,1,1,4)
cls = np.reshape(outputs[0], (2,))
reg = np.reshape(outputs[1], (4,))
print(cls[0] + cls[1] )
print(cls)
print(outputs[0])