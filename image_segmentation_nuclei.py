"""
Project-Image Segmentation of nuclei.
A model for semantic segmentation for images containing cell neuclei.
"""
#%%
#1. Import packages
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, os, datetime
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
#%%
train_filepath = "C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment4\\data-science-bowl-2018\\data-science-bowl-2018-2\\train"
test_filepath= "C:\\Users\\IT\\Desktop\\AI lesson\\DEEP LEARNING\\hands-on\\Assessment4\\data-science-bowl-2018\\data-science-bowl-2018-2\\test"
images = []
masks = []
test_images = []
test_masks = []
#%%
#2. Load images
image_path = os.path.join(train_filepath,'inputs')
for img in os.listdir(image_path):
    #Get the full path of the image file
    full_path = os.path.join(image_path,img)
    #Read the image file based on the full path
    img_np = cv2.imread(full_path)
    #Convert the image from bgr to rgb
    img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    #Resize the image into 128x128
    img_np = cv2.resize(img_np,(128,128))
    #Place the image into the empty list
    images.append(img_np)

#3. Load masks
mask_path = os.path.join(train_filepath,'masks')
for mask in os.listdir(mask_path):
    #Get the full path of the mask file
    full_path = os.path.join(mask_path,mask)
    #Read the mask file as a grayscale image
    mask_np = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
    #Resize the image into 128x128
    mask_np = cv2.resize(mask_np,(128,128))
    #Place the mask into the empty list
    masks.append(mask_np)
#%%
# For the test dataset
test_image_path = os.path.join(test_filepath, 'inputs')
for img in os.listdir(test_image_path):
    full_path = os.path.join(test_image_path, img)
    img_np = cv2.imread(full_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (128, 128))
    test_images.append(img_np)

# Load test masks
test_mask_path = os.path.join(test_filepath, 'masks')
for mask in os.listdir(test_mask_path):
    full_path = os.path.join(test_mask_path, mask)
    mask_np = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    mask_np = cv2.resize(mask_np, (128, 128))
    test_masks.append(mask_np)

# %%
#4. Convert the list of np array into a full np array
images_np = np.array(images)
masks_np = np.array(masks)
#5. Data preprocessing
#5.1. Expand the mask dimension to include the channel axis
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#5.2. Convert the mask value into just 0 and 1
converted_masks_np = np.round(masks_np_exp/255)
#5.3. Normalize the images pixel value
normalized_images_np = images_np/255.0
#%%
#6. For the test dataset
# Convert test data to numpy arrays
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

# Preprocessing for test data
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)
converted_test_masks_np = np.round(test_masks_np_exp / 255)
normalized_test_images_np = test_images_np / 255.0
# %%
#7. Convert the numpy array into tensorflow tensors
# Convert the numpy arrays into TensorFlow tensors for train dataset
train_images_tensor = tf.data.Dataset.from_tensor_slices(normalized_images_np)
train_masks_tensor = tf.data.Dataset.from_tensor_slices(converted_masks_np)

# Convert the numpy arrays into TensorFlow tensors for test dataset
test_images_tensor = tf.data.Dataset.from_tensor_slices(normalized_test_images_np)
test_masks_tensor = tf.data.Dataset.from_tensor_slices(converted_test_masks_np)
#%%
#8. Combine features and labels together to form a zip dataset
# Combine features and labels for train dataset
train = tf.data.Dataset.zip((train_images_tensor, train_masks_tensor))

# Combine features and labels for test dataset
test = tf.data.Dataset.zip((test_images_tensor, test_masks_tensor))

# %%
#Convert this into prefetch dataset
BATCH_SIZE = 64
sample_image = test_images[0]
sample_mask = test_masks[0]
# Shuffle, batch, and prefetch the training dataset
train_dataset = train.shuffle(buffer_size=len(images_np)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
# Batch the testing dataset
test_dataset = test.batch(BATCH_SIZE)
# %%
# Print an example from the training dataset
for image, mask in train_dataset.take(1):
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
# %%
#9. Model development
"""
The plan is to apply transfer learning by using a pretrained model as the feature extractor.
Then, we will proceed to build our own upsampling path with the tensorflow_example module we just imported + other default keras layers.
"""
#9.1. Use a pretrained model as feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
base_model.summary()
plot_model(base_model,show_shapes=True,show_layer_names=True)
# %%
#9.2. Specify the layers that we need as outputs for the feature extractor
layer_names = [
    "block_1_expand_relu",      #64x64
    "block_3_expand_relu",      #32x32
    "block_6_expand_relu",      #16x16
    "block_13_expand_relu",     #8x8
    "block_16_project"          #4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#9.3. Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#9.4. Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),        #4x4  --> 8x8
    pix2pix.upsample(256,3),        #8x8  --> 16x16
    pix2pix.upsample(128,3),        #16x16 --> 32x32
    pix2pix.upsample(64,3)          #32x32 --> 64x64
]

#9.5. Define a function for the unet creation.
def unet(output_channels:int):
    """
    We are going to use functional API to connect the downstack and upstack properly
    """
    #(A) Input layer
    inputs = keras.Input(shape=[128,128,3])
    #(B) Down stack (Feature extractor)
    skips = down_stack(inputs)
    x = skips[-1]       #This is the output that will progress until the end of the model
    skips = reversed(skips[:-1])

    #(C) Build the upsampling path
    """
    1. Let the final output from the down stack flow through the up stack
    2. Concatenate the output properly by following the structure of the U-Net
    """
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])

    #(D) Use a transpose convolution layer to perform one last upsampling. This convolution layer will become the output layer as well.
    last = keras.layers.Conv2DTranspose(output_channels,kernel_size=3,strides=2,padding='same')     #64x64 --> 128x128
    outputs = last(x)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model
# %%
#9.6. Create the U-Net model by using the function
OUTPUT_CLASSES = 3
model = unet(OUTPUT_CLASSES)
model.summary()
keras.utils.plot_model(model)
# %%
#10. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
# %%
#11. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]       #equivalent to tf.expand_dims()
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()
#%%
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ["Input Image","True Mask","Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for images,masks in train_dataset.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

# %%
#12. Create a custom callback function to display results during model training
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch #{}\n'.format(epoch+1))
#%%
#Create tensorboard
base_log_path = r"tensorboard_logs\nuclei_image_segmentation"
if not os.path.exists(base_log_path):
    os.makedirs(base_log_path)
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
#%%
#13. Model training
#Implement the EarlyStopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=3)
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(images) //BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = len(mask) // BATCH_SIZE
history = model.fit(train_dataset,
                    validation_data=test_dataset,                
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    callbacks=[DisplayCallback(), tb, keras.callbacks.TensorBoard(log_dir=log_path),
                               keras.callbacks.EarlyStopping(patience=3)],)
# %%
#14. Model deployment
show_predictions(test_dataset,3)
# %%
#15. Save the model
save_path = os.path.join("save_model","nuclei_image_segmentation_model.h5")
model.save(save_path)

# %%
