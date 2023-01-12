import tensorflow as tf
import os
import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt
from keras_unet_collection import losses
from skimage.transform import resize
from tensorflow.keras.metrics import MeanIoU

def iou(y_true, y_pred, dtype=tf.float32):
    return 1 - losses.iou_seg(y_true, y_pred, dtype=tf.float32)   

class SegEvaluate:

    def __init__(self, test_dir, model_dir):
        self.test_dir = test_dir
        self.model_dir = model_dir
        
        self.model = tf.keras.models.load_model(
            model_dir,
            custom_objects = {
                'dice_coef': losses.dice_coef,
                'iou': iou
            }
        )

        self.x_test, self.y_test = self.get_data(test_dir)

    def test_eval(self):
        loss, acc, dice, iou = self.model.evaluate(
            self.x_test, self.y_test, verbose=1
        )
        print(loss, acc, dice, iou)

    def test_pred(self, ind):
        test_preds = self.model.predict(self.x_test)
        preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)
        for i in range(ind):
            fig, axs = plt.subplots(1, 3)
            fig.set_figheight(20)
            fig.set_figwidth(10)
            test_img = preds_test_thresh[i, :, :, 0]
            axs[0].imshow(test_img, cmap='gray')
            axs[0].set_title('Pred mask')
            axs[1].imshow(self.y_test[i][:,:,0], cmap='gray')
            axs[1].set_title('True mask')
            axs[2].imshow(cv2.cvtColor(self.x_test[i], cv2.COLOR_BGR2RGB))
            axs[2].set_title('Image')
            plt.show()

    def get_data(self, data_dir):
        print('Getting data...')
        print(data_dir, 'path exists:', os.path.exists(data_dir))
        ids = next(os.walk(data_dir))[1]
        print('IDs in {0}: {1}'.format(data_dir, ids))

        x = np.zeros((len(ids), 128, 128, 3), dtype=np.uint8)
        y = np.zeros((len(ids), 128, 128, 1), dtype=bool)
        for n, name in enumerate(ids):
            img_dir = os.path.join(data_dir, name, 'Resized_Images')
            img_file = os.path.join(img_dir, os.listdir(img_dir)[0])
            img = cv2.imread(img_file)
            img = resize(img, (128, 128), mode='constant', preserve_range=True)
            x[n] = img

            mask = np.zeros((128, 128, 1), dtype=bool)
            mask_dir = os.path.join(data_dir, name, 'Resized_Masks')
            mask_file = os.path.join(mask_dir, os.listdir(mask_dir)[0])
            mask_read = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask_read = np.expand_dims(resize(mask_read, (128, 128), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_read)
            y[n] = mask
        return x, y


if __name__ == '__main__':
    print('Running seg_evaluate.py...')
    seg_evaluate = SegEvaluate(*sys.argv[2:4])
    if sys.argv[1] == 'pred':
        seg_evaluate.test_pred(int(sys.argv[4]))
    elif sys.argv[1] == 'eval':
        seg_evaluate.test_eval()
        





















