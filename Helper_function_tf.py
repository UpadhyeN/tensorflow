
# ploting loss curves
# ploting a loss curve
def plot_loss_curves(history):
  loss = history.history['loss']
  val_loss = history.history["val_loss"]
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))
# plot the loss
  plt.plot(epochs,loss,label="training_loss")
  plt.plot(epochs,val_loss,label="val_loss")
  plt.legend()
# plot the accuracy
# plt.figure() to create new figure
  plt.figure()
  plt.plot(epochs,accuracy,label="training_accuracy")
  plt.plot(epochs,val_accuracy,label="val_accuracy")
  plt.legend()
  
  
# lets create_model() function to create a model from a URL.
def create_model(model_url,num_classes=10):
  """
  takes TensorFlow Hub URL and creates a Keras Sequential model with it.
  Args:
  model_url (str) : A tensorflow Hub feature extraction URL.
  num_classes (int) : number of output neurons in the output layer,
  should be equal to number of target classes, default 10,

  Returns:
  An uncomplied Keras Seq model with model_url as feature extractor layer and Dense output laery with num_classes output neurons
  
  """
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False,
                                           name="feature_extraction_layer",
                                           input_shape=IMAGE_SHAPE+(3,))
   # freeze
  model = tf.keras.Sequential([feature_extractor_layer,
                             layers.Dense(num_classes,activation="softmax",name="output_layer")])


  return model



# create function to preprocess the image
def load_prep_function(filename,img_shape=224):
  '''
  reads an image from filename and truns into a tensor and reshape.
  '''
  # read in image
  img = tf.io.read_file(filename)
  # decode the read file into a tensor
  img = tf.image.decode_jpeg(img)
  #resize
  img = tf.image.resize(img,size=[img_shape,img_shape])
  # rescale the image
  img = img/255.
  return img

def pred_and_plot(model,filename,class_names=class_names):
  """
  imports an image located at filename, makes a predictions with model.
  and plot the model
  """
  img = load_prep_function(filename)

  pred = model.predict(tf.expand_dims(img,axis=0))
  pred = list(tf.round(pred))
  pred = pred[0]
  pred = list(pred.numpy())
  print(pred)
  index = pred.index(1)
  #pred_class = class_names[int(tf.round(pred))]
  pred_class = class_names[index]
  # plot the image and 
  
  plt.imshow(img)
  plt.title(f'Predictions:{pred_class}')
  plt.axis(False);
