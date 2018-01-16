import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct

import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *

#------------------BASIC SETUP--------------------------------#

data_dir = '/storage/work/tks5206/input/'
train_bson_path = os.path.join(data_dir, 'train.bson')
num_train_products = 7069896

test_bson_path = os.path.join(data_dir, 'test.bson')
num_test_products = 1768182

#------------CREATE LOOKUP TABLES----------------------------#

categories_path = os.path.join(data_dir, 'category_names.csv')
categories_df = pd.read_csv(categories_path, index_col='category_id')

#maps caterogy_id to integer, this is what we will use to one-hot encode the labels
categories_df['category_idx'] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv('categories.csv')
print(categories_df.head())

#-------------CREATE DICTIONARIES-----------------------------#

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

cat2idx, idx2cat = make_category_tables()

#Test if it works
print(cat2idx[1000012755], idx2cat[4])

#---------------READ THE BSON FILES-----------------------------------#

def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, 'rb') as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack('<i', item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item['_id']
            num_imgs = len(item['imgs'])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item['category_id']]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ['num_imgs', 'offset', 'length']
    if with_categories:
        columns += ['category_id']

    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'product_id'
    df.columns = columns
    df.sort_index(inplace=True)
    return df

train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)

print(train_offsets_df.head())

train_offsets_df.to_csv('train_offsets.csv')

#How many products
print(len(train_offsets_df))

#How many categories
print(len(train_offsets_df['category_id'].unique()))

#How many images in total
print(train_offsets_df['num_imgs'].sum())

#----------------CREATE RANDOM TRAIN/VALIDATION SPLIT-------------------#

def make_val_set(df, split_percentage = 0.2, drop_percentage = 0.0):
    #Find the product_id for each category
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            #Randomly remove products to make the dataset smaller
            keep_size = int(len(product_ids) * (1.0 - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            #Randomly choose the products that become part of the validation set
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []


            #Create a new row for each image
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, 'num_imgs']):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                    pbar.update()

    columns = ['product_id', 'category_idx', 'img_idx']
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df


#Create 80/20 split also drop 90% of products to make more manageable
train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, drop_percentage=0.0)

print(train_images_df.head())

print(val_images_df.head())

print('Number of training images:', len(train_images_df))
print('Number of validation images:', len(val_images_df))
print('Total images:', len(train_images_df) + len(val_images_df))

#Are all categories represented in the train/val split?
print(len(train_images_df['category_idx'].unique()), len(val_images_df['category_idx'].unique()))

#Save lookup tables as csv so we do not need to repeat procedure again
train_images_df.to_csv('train_images.csv')
val_images_df.to_csv('val_images.csv')

#-------------------------GENERATOR-----------------------------------------#

#Uncomment these to Load in the data from part 1 so we do not need to run again
#categories_df = pd.read_csv('categories.csv', index_col=0)
#cat2idx, idx2cat = make_category_tables()
#train_offsets_df = pd.read_csv('train_offsets.csv', index_col=0)
#train_images_df = pd.read_csv('train_images.csv', index_col=0)
#val_images_df = pd.read_csv('val_images.csv', index_col=0)


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.data_utils import Sequence

class Iterator(Sequence):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180),
                 with_labels=True, batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

train_bson_file = open(train_bson_path, 'rb')

import threading
lock = threading.Lock()

num_classes = 5270
num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 128

#Tip: Use ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, num_classes, train_datagen, lock, batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df, num_classes, val_datagen, lock, batch_size=batch_size, shuffle=True)

next(train_gen)

bx, by = next(train_gen)

#plt.imshow(bx[-1].astype(np.uint8))

cat_idx = np.argmax(by[-1])
cat_id = idx2cat[cat_idx]
print(categories_df.loc[cat_id])

bx, by = next(val_gen)

#plt.imshow(bx[-1].astype(np.uint8))

cat_idx = np.argmax(by[-1])
cat_id = idx2cat[cat_idx]
print(categories_df.loc[cat_id])

#-------------------------------EDIT by TEJAS SHAHPURI--------------------------#

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, GlobalAveragePooling2D

from inception_v4 import create_inception_v4
from inception_resnet_v2 import InceptionResNetV2

class Models:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.model = Sequential()
    
    def vgg16(self):
        base_model = VGG16(include_top=False, weights='imagenet',
                           input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def vgg19(self):
        base_model = VGG19(include_top=False, weights='imagenet',
                            input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))
            
    def resnet50(self):
        base_model = ResNet50(include_top=False, weights='imagenet',
                              input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def inceptionV3(self):
        base_model = InceptionV3(include_top=False, weights='imagenet',
                                 input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='relu'))
        self.model.add(Dense(self.classes, activation='softmax'))

    def InceptionResNet_V2(self):
        base_model = InceptionResNetV2(include_top=True, weights='imagenet',
                                        input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='relu'))
        self.model.add(Dense(self.classes, activation='softmax'))
            
    def compile(self, optimizer):
        print(self.model.summary())
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                             
    def load_weights(self, path):
        self.model.load_weights(path)
                             
    def save_weights(self, path):
        self.model.save_weights(path)
                             
    def get_model(self):
        return self.model

    def inceptionV4(self):
        base_model = create_inception_v4(nb_classes=self.classes)
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='relu'))
        self.model.add(Dense(self.classes, activation='softmax'))



#-----------------------TRAINING------------------------------------------------#

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from collections import defaultdict
from keras.models import Model

import params

import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import RMSprop, SGD
from multi_gpu_model import MultiGPUModel

#model = Sequential()
#model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(180, 180, 3)))
#model.add(MaxPooling2D())
#model.add(Conv2D(64, 3, padding="same", activation="relu"))
#model.add(MaxPooling2D())
#model.add(Conv2D(128, 3, padding="same", activation="relu"))
#model.add(MaxPooling2D())
#model.add(GlobalAveragePooling2D())
#model.add(Dense(num_classes, activation="softmax"))

#------------------------Edit 1--------------------------------------------------#
callbacks = [ModelCheckpoint(filepath='/storage/work/tks5206/weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
            TensorBoard(log_dir='/storage/work/tks5206/', 
                        histogram_freq=0, 
                        batch_size=batch_size),
            ReduceLROnPlateau(factor=0.5,
                               patience=2,
                               verbose=1,
                               epsilon=1e-4),
            EarlyStopping(min_delta=1e-4,
                           patience=4,
                           verbose=1)]

#with tf.device('/cpu:0'):
#            models = Models(input_shape=(180, 180, 3), classes=num_classes)
#            tmodel = models.inceptionV3()

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
#top_model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=base_model.output_shape)
#top_model.add(Dense(num_classes, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(GlobalAveragePooling2D())
#top_model.add(Dropout(0.5))
#top_model.add(Dense(num_classes, activation='softmax', input_shape=base_model.output_shape))
#top_model.load_weights(top_model_weights_path)

with tf.device('/cpu:0'):
        model = Model(inputs=base_model.inputs, outputs=model.outputs)

parallel_model = MultiGPUModel(model, 4)
    #if params.base_model == 'vgg16':
#models.InceptionResNet_V2()
    #elif params.base_model == 'vgg19':
#models.vgg19()
#elif params.base_model == 'resnet50':
#models.resnet50()
#elif params.base_model == 'inceptionV3':
#    models.inceptionV3()
#else:
#    print('Uknown base model')
#    raise SystemExit

#parallel_model.load_weights('/storage/work/tks5206/weights/best_weights.hdf5')

parallel_model.compile(optimizer=SGD(lr=0.1, momentum=0.9, decay=0.1, nesterov=True),
                        loss="categorical_crossentropy", 
                        metrics=["accuracy"])

#models.compile(optimizer='sgd',
#              loss="categorical_crossentropy",
#              metrics=["accuracy"])

#model = parallel_model.get_model()

#model.fit_generator(train_gen,
#                    steps_per_epoch= (float(len(train_images_df))/ float(batch_size))/100,
#                    epochs=5,
#                    verbose=1,
#                    validation_data=val_gen,
#                    validation_steps= float((len(val_images_df)) / float(batch_size))/100,
#                    callbacks=callbacks,
#                    workers = 8)

#------------------------Original without Edit1--------------------------------------------------#

# To train the model:
parallel_model.fit_generator(train_gen,
                    steps_per_epoch = (num_train_images//batch_size)/10,   #num_train_images // batch_size,
                    epochs = 5,
                    verbose=1,
                    validation_data = val_gen,
                    validation_steps = (num_val_images//batch_size)/10,  #num_val_images // batch_size,
                    callbacks=callbacks,
                    workers = 8)


# To train the model:
#model.fit_generator(train_gen,
#                    steps_per_epoch = 10,   #num_train_images // batch_size,
#                    epochs = 20,
#                    verbose=1,
#                    validation_data = val_gen,
#                    validation_steps = 10,  #num_val_images // batch_size,
#                    callbacks=callbacks,
#                    workers = 8)

# To evaluate on the validation set:
#model.evaluate_generator(val_gen,
 #                        steps= float((len(val_images_df)) / float(batch_size))/100,          #num_val_images // batch_size
 #                        workers=8)

parallel_model.evaluate_generator(val_gen,
                         steps= (num_val_images//batch_size)/10,          #num_val_images // batch_size
                         workers=8)

#------------------------PREDICTIONS--------------------------------------------------#

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

submission_df = pd.read_csv(data_dir + 'sample_submission.csv')
submission_df.head()

test_datagen = ImageDataGenerator()
data = bson.decode_file_iter(open(test_bson_path, 'rb'))


with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d['_id']
        num_imgs = len(d['imgs'])

        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d['imgs'][i]['picture']

            #Load and preprocess the image
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            #Add the image to the batch
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]['category_id'] = idx2cat[cat_idx]
        pbar.update()

submission_df.to_csv('my_submission.csv.gz', compression='gzip', index=False)
