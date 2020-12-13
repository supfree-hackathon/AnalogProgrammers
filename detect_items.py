import imutils

# from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
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


def detect_items(image=None, size=(200, 150), min_conf=0.9, visualize=-1):
    WIDTH = 600
    PYR_SCALE = 1.5
    WIN_STEP = 16
    ROI_SIZE = size
    INPUT_SIZE = (224, 224)

    # load our network weights from disk
    print("[INFO] loading network...")
    # model = ResNet50(weights="imagenet", include_top=True)

    model = keras.models.load_model("sup_free_model.h5")
    print('model loaded')
    print(model)

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
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
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
    print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))
    rois = np.array(rois, dtype="float32")
    print("[INFO] classifying ROIs...")
    start = time.time()
    print(rois)
    print(rois.shape)
    rois = rois / 255.0
    print(rois.shape)
    preds = model.predict(rois)
    end = time.time()
    print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))
    print(preds)
    print(type(preds))
    print(dir(preds))


    # preds = imagenet_utils.decode_predictions(preds, top=1)


    # labels = {}

    # loop over the predictions
    # for (i, p) in enumerate(preds):
    #     # grab the prediction information for the current ROI
    #     (imagenetID, label, prob) = p[0]
    #     # filter out weak detections by ensuring the predicted probability
    #     # is greater than the minimum probability
    #     if prob >= args["min_conf"]:
    #         # grab the bounding box associated with the prediction and
    #         # convert the coordinates
    #         box = locs[i]
    #         # grab the list of predictions for the label and add the
    #         # bounding box and probability to the list
    #         L = labels.get(label, [])
    #         L.append((box, prob))
    #         labels[label] = L
    #
    #


if __name__ == '__main__':
    detect_items(image=cv2.imread('PET_bottle.jpg'))
