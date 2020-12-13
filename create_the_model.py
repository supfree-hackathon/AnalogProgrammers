import tensorflow as tf

import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator

    # creates a data generator object that transforms images
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # vertical_flip=True,
        fill_mode='nearest')

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    K = 0

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #  'dog', 'frog', 'horse', 'ship', 'truck']
    # class_ids = [9,10,16,28,61]
    # class_names = ['bottles', 'bowls', 'cans', 'cups', 'plates']
    class_ids = [9, 28, 61]
    class_names = ['bottles', 'cups', 'plates']

    new_train_images = []
    new_train_labels = []
    new_test_images = []
    new_test_labels = []

    for IMG_INDEX, i in enumerate(train_labels):
        if i[0] in class_ids:
            new_train_images.append(train_images[IMG_INDEX])
            new_train_labels.append(np.array([class_ids.index(train_labels[IMG_INDEX])]))

            test_img = train_images[IMG_INDEX]
            img = image.img_to_array(test_img)  # convert image to numpy arry
            img = img.reshape((1,) + img.shape)  # reshape image

            i = 0
            if K:
                for batch in datagen.flow(img, save_prefix='test',
                                          save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
                    if i >= K:  # show 4 images
                        break
                    new_train_images.append(image.img_to_array(batch[0]))
                    new_train_labels.append(np.array([class_ids.index(train_labels[IMG_INDEX])]))
                    i += 1
                    # plt.figure(i)
                    # print(new_train_labels[-1])
                    # print(new_train_labels[0])
                    # plot = plt.imshow(new_train_images[-1])

    print('complete with train')

    for IMG_INDEX, i in enumerate(test_labels):
        if i[0] in class_ids:
            new_test_images.append(test_images[IMG_INDEX])
            new_test_labels.append(np.array([class_ids.index(test_labels[IMG_INDEX])]))

            test_img = test_images[IMG_INDEX]
            img = image.img_to_array(test_img)  # convert image to numpy arry
            img = img.reshape((1,) + img.shape)  # reshape image

            i = 0
            if K:
                for batch in datagen.flow(img, save_prefix='test',
                                          save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
                    if i >= K:  # show 4 images
                        break
                    new_test_images.append(image.img_to_array(batch[0]))
                    new_test_labels.append(np.array([class_ids.index(test_labels[IMG_INDEX])]))
                    i += 1
                    # plt.figure(i)
                    # print(new_train_labels[-1])
                    # print(new_train_labels[0])
                    # plot = plt.imshow(new_train_images[-1])

    print('complete with test')
    randomized_train_images = []
    randomized_train_labels = []
    from random import randint

    while len(new_train_labels):
        i = randint(0, len(new_train_labels) - 1)
        randomized_train_images.append(new_train_images[i])
        del new_train_images[i]
        randomized_train_labels.append(new_train_labels[i])
        del new_train_labels[i]

    # train_images = np.array(new_train_images)
    # train_labels = np.array(new_train_labels)

    train_images = np.array(randomized_train_images)
    train_labels = np.array(randomized_train_labels)

    test_images = np.array(new_test_images)
    test_labels = np.array(new_test_labels)

    del new_train_images
    del new_train_labels
    del new_test_images
    del new_test_labels
    del class_ids

    print('train_labels : ')
    print(set(i[0] for i in train_labels))
    print(len(train_images))
    print(len(train_labels))
    print('test_labels : ')
    print(set(i[0] for i in test_labels))
    print(len(test_images))
    print(len(test_labels))

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.summary()  # let's have a look at our model so far
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(5))
    model.summary()

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=7,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    model.save('sup_free_model')