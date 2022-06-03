# 1. load pre-trained imeagenet date Vgg16 model without fully-connected layer
#include_top = False --> fully connected layer 
from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)

# 2.Freeze Base Model
# ImageNet dataset does not get destroyed in the initial training.

base_model.trainable = False

# 3. Add Layers to Model
# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

# 4. Compile Model
model.compile(loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# 5. Augment the Data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
                samplewise_center=True,  # set each sample mean to 0
                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range=0.1,  # Randomly zoom image
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False,
                                   )
datagen_valid = ImageDataGenerator(samplewise_center=True,)

# 6.Loading the Data
#load images directly from folders using Keras' flow_from_directory function.'
# size our images to match the model: 244x244 pixels with 3 channels.

# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    "data/fruits/train/",
    target_size=(244,244),
    color_mode="rgb",
    class_mode="categorical",
)
# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    "data/fruits/valid/",
    target_size=(244,244),
    color_mode="rgb",
    class_mode="categorical",
)

# 7. Train the Model
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=20)

# 8. Unfreeze Model for Fine Tuning
# Unfreeze the base model
base_model.trainable = True

# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = 0.00001),
              loss = 'categorical_crossentropy' , metrics = ['accuracy'])

#re fitting
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=12)

# 8. Evaluate the Model
# first value is loss, and the second value is accuracy
model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)

# 9. Run the Assessment
from run_assessment import run_assessment

run_assessment(model, valid_it)
