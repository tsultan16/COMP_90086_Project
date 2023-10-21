import random, os, time
from pathlib import Path
import numpy  as np
import csv

# do this before importing tensorflow to get rid of annoying warning messages
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import ResNet50V2
import matplotlib.pyplot as plt

"""
    
    Just a library of utitlity functions.

    
    Code Author: Tanzid Sultan (ID# 1430660)    

"""



"""
    get file paths of all left-right images
"""
def load_image_paths(path='COMP90086_2023_TLLdataset/'):

    # read left-right pairs image ids from csv file
    file = open(path+'train.csv')
    csvreader = csv.reader(file)
    header = next(csvreader) # skip header
    pairs = []
    for row in csvreader:
        pairs.append(row)
    file.close()

    # convert image ids to filepaths
    left_paths = []
    right_paths = []
    for pair in pairs:
        left_paths.append(path + 'train/left/' + pair[0] + '.jpg')
        right_paths.append(path + 'train/right/' + pair[1] + '.jpg')

    return left_paths, right_paths



"""
    generates triplets of (anchor,positive,negative) with randomly chosen negatives, can specify the number of triplets per anchor-positive pair with a different negatives
"""
def generate_triplets_randomly(anchors, positives, num_negatives):
    
    # first combine the anchor and positive image sets and shuffle it up
    negatives = anchors + positives
    np.random.RandomState(seed=32).shuffle(negatives)
    
    # now, randomly sample num_negatives+2 negative images for each anchor
    # (we sample an extra two images in case there are duplicates
    # of the positive or anchor image in the negative images set)
    triplets = []
    for i in range(0, len(anchors)):
        # randomly sample num_negatives+2 images without replacement
        negative_set = random.sample(negatives, num_negatives+2)
        # remove any duplicates
        negative_set = [img for img in negative_set if img != anchors[i] and img != positives[i]] 
        negative_set = negative_set[:num_negatives]

        for negative_img in negative_set:
            triplets.append((anchors[i], positives[i], negative_img))    

    # shuffle the triplet set
    np.random.RandomState(seed=2).shuffle(triplets)

    triplet_anchors = [t[0] for t in triplets]
    triplet_positives = [t[1] for t in triplets]
    triplet_negatives = [t[2] for t in triplets]

    return (triplet_anchors, triplet_positives, triplet_negatives)


# instead of picking negative images randomly, can use the distance metric to find the most similar negative images to the anchor and positive
# these are the so-called "hard negatives"
#def generate_triplets_randomly_hard_negatives():
#    ...


"""
    create tensorflow triplet dataset
"""
def create_tf_dataset_triplets(anchors, positives, negatives, batch_size):
    # create tensorflow datasets
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchors)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positives)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negatives)
    
    # create a triplet dataset
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)


    # map filenames to preprocessed images
    dataset = dataset.map(preprocess_triplets) #, num_parallel_calls=tf.data.AUTOTUNE) 
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 

    return dataset
        

"""
    load image jpeg image from file and rescale pixel intensities
"""
def preprocess_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


""" 
    preprocess a triplet of images given their filenames
"""
def preprocess_triplets(anchor, positive, negative):
    return (preprocess_image(anchor), preprocess_image(positive), preprocess_image(negative))


"""
    split into training-validation datasets
"""
def train_val_split(path='COMP90086_2023_TLLdataset/', split_ratio=0.8, num_negatives=20, batch_size=32):
    # get paths of left-right pair images
    anchor_paths, positive_paths = load_image_paths(path=path)
    image_count = len(anchor_paths)

    # split into training and validation sets
    num_train = int(split_ratio * image_count)
    anchor_paths_train = anchor_paths[:num_train]
    anchor_paths_val = anchor_paths[num_train:]
    positive_paths_train = positive_paths[:num_train]
    positive_paths_val = positive_paths[num_train:]

    # training set triplets
    (anchor_train, positive_train, negative_train) = generate_triplets_randomly(anchor_paths_train, positive_paths_train, num_negatives=num_negatives)
    # validation set triplets
    (anchor_val, positive_val, negative_val) = generate_triplets_randomly(anchor_paths_val, positive_paths_val, num_negatives=num_negatives)
    
    # create tensorflow datasets
    triplets_train = create_tf_dataset_triplets(anchor_train, positive_train, negative_train, batch_size=batch_size)
    triplets_val = create_tf_dataset_triplets(anchor_val, positive_val, negative_val, batch_size=batch_size)

    return triplets_train, triplets_val, anchor_paths_train, positive_paths_train, anchor_paths_val, positive_paths_val


"""
    visualize triplets from dataset
"""
def visualize(triplets_dataset, num_triplets=3):
    
    anchor, positive, negative_set = list(triplets_dataset.take(1).as_numpy_iterator())[0]

    def show(ax, image):
        ax.imshow(image)
        ax.axis('off')
        
    print(anchor[0].shape)


    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(num_triplets, 3)
        
    for i in range(num_triplets):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative_set[i])

