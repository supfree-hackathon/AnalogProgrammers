import imutils

# from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
from PIL import Image
# from scikit import transform

import argparse
import imutils
import time
import cv2


def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def detect_items(image=None, size=(200, 200), min_conf=3.3, visualize=-1):
    WIDTH = 600
    PYR_SCALE = 1.1
    WIN_STEP = 4
    ROI_SIZE = size
    INPUT_SIZE = (32, 32)
    class_names = {0 :'bottles', 1: 'cups', 2: 'plates'}
    # load our network weights from disk
  # print("[INFO] loading network...")
    # model = ResNet50(weights="imagenet", include_top=True)

    model = keras.models.load_model("sup_free_model.h5")
    # print('model loaded')
    # print(model)

    orig = image
    orig = imutils.resize(orig, width=WIDTH)
    (H, W) = orig.shape[:2]

    pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

    rois = []
    locs = []

    start = time.time()

    # loop gia kathe epipedo ths pyramidas
    for image in pyramid:
        scale = W / float(image.shape[1])
        # loop gia kathe seiromeno parathyro
        # for (x, y, roiOrig) in sliding_window(image, image.shape[0]//8, ROI_SIZE):
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            # print(roi)
            # print(roi.shape)
            # input()
            roi = img_to_array(roi)
            # print(roi)
            # print(type(roi))
            # print(roi.shape)
            roi = preprocess_input(roi)
            # print(roi)
            # print(roi.shape)
            # print(type(roi))
            # roi = array_to_img(roi)

            # print(roi)
            # print(type(roi))
            # print(roi.shape)
            # input('->')



            rois.append(roi)
            locs.append((x, y, x + w, y + h))

            if visualize > 0:
                clone = orig.copy()
                cv2.rectangle(clone, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                # show the visualization and current ROI
                cv2.imshow("Visualization", clone)
                cv2.imshow("ROI", roiOrig)
                cv2.waitKey(0)

    end = time.time()
  # print("[ INFO ] looping over pyramid/windows took {:.5f} seconds".format(end - start))
    # print(len(rois))
    # rois = [np.array(array_to_img(roi).getdata()).reshape(roi.size[0], roi.size[1], 3) for roi in rois]
    # print(type(rois[0]))
    rois = np.array(rois)
    # print(rois[0].shape)
  # print("[INFO] classifying ROIs...")
    start = time.time()
    start = time.time()
    # print(rois)
    # print(rois.shape)
    rois = rois / 255.0
    # print(rois.shape)
    preds = model.predict(rois)
    end = time.time()
  # print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))
    # print(preds)
    # print(type(preds))
    # print(dir(preds))


    # preds = imagenet_utils.decode_predictions(preds, top=1)


    # labels = {}

    # loop over the predictions
    labels = {}
    for (i, p) in enumerate(preds):
    #     # grab the prediction information for the current ROI
        p = list(p)
        prob =max(p)
        label = p.index(prob)
        if prob >= min_conf:
            print(label)
            box = locs[i]
            L = (box, prob)
            # print('L', L)
            try:
                labels[label].append(L)
            except:
                labels[label] = [L]

    # loop over the labels for each of detected objects in the image
    clone = orig.copy()
    for label in class_names.keys():
        if label not in labels.keys():
            break
        # clone the original image so that we can draw on it
      # print("[INFO] showing results for '{}'".format(label))

        # loop over all bounding boxes for the current label
        # print(labels[label])
        print('-----------------------------------')
        # for (box, prob) in labels[label]:
        #   # print(prob)
        #     # draw the bounding box on the image
        #     (startX, startY, endX, endY) = box
        #     cv2.rectangle(clone, (startX, startY), (endX, endY),
        #                   (0, 255, 0), 2)
        # show the results *before* applying non-maxima suppression, then
        # clone the image again so we can display the results *after*
        # applying non-maxima suppression
        # cv2.imshow("Before", clone)
        # clone = orig.copy()
        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        # loop over all bounding boxes that were kept after applying
        # non-maxima suppression
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            # draw the bounding box and label on the image
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, class_names.get(label)+str(labels[label][i][1]), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        # show the output after apply non-maxima suppression
    return clone


def test():
    detect_items(image=cv2.imread('PET_bottle_1.jpg'), visualize=-1)
if __name__ == '__main__':
    # detect_items(image=cv2.imread('PET_bottle_1.jpg'), visualize=-1)

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here

        new_frame = detect_items(image=frame, visualize=-1)

        # Display the resulting frame
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # model = keras.models.load_model("sup_free_model.h5")
    # np_image = Image.open('PET_bottle.jpg')
    # np_image = np.array(np_image).astype('float32') / 255
    # np_image = transform.resize(np_image, (256, 256, 3))
    # np_image = np.expand_dims(np_image, axis=0)

    # img_width, img_height = 32, 32
    # img = keras.preprocessing.image.load_img('PET_bottle.jpg', target_size=(img_width, img_height))
    # img = img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    #
    # # image = cv2.imread('PET_bottle.jpg')
    # # image = image.reshape((1,) + image.shape)
    # # image = image/255.0
    # # image = image.img_to_array(image)
    # # preds = model.predict([image])
    #
    # preds = model.predict(img)
    # print(preds)