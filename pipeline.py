import albumentations as A
import cv2
import json
import numpy as np
import os
import seaborn as sns
import sys
import tensorflow as tf
import keras
import shutil

from keras_unet_collection import losses, models
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import KFold
from tensorflow import float64
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

shape = (128, 128)
batch_size = 8
K = 10  # denotes cross-validation folds
ms = 3  # denotes pyplot 'marksersize'
save_dir = '/content/saved_models' # specific to Colab instance
sns.set_theme()

model_dict = {
    'unet_2d': models.unet_2d,
    # 'att_unet_2d': models.att_unet_2d
}

def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 50
    if epoch % decay_step == 0 and epoch > 1:
        return lr * decay_rate
    return lr 

class SegPipeline:

    def __init__(self, backbone, weights, model_name, train_dir, test_dir, test_save_dir,
                 lr, epochs, filters):
        self.backbone = backbone
        self.weights = weights
        self.model_name = model_name
        self.model = model_dict[model_name]
        self.lr = lr
        self.epochs = epochs
        self.filters = filters
        self.filter_num = []
        for i in range(filters):
            self.filter_num.append(64*(2**i))

        self.x_train, self.y_train = self.get_data(train_dir)
        self.x_test, self.y_test = self.get_data(test_dir)

        self.save_path = os.path.join(save_dir, model_name + '_{:.0e}_{:1}'.format(self.lr, self.filters))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.test_save_dir = test_save_dir
        if not os.path.exists(self.test_save_dir):
            os.makedirs(self.test_save_dir)

    def get_data(self, data_dir):
        print('Getting data...')
        print(data_dir, 'path exists:', os.path.exists(data_dir))
        ids = next(os.walk(data_dir))[1]
        print('IDs in {0}: {1}'.format(data_dir, ids))

        x = np.zeros((len(ids), shape[0], shape[1], 3), dtype=np.uint8)
        y = np.zeros((len(ids), shape[0], shape[1], 1), dtype=bool)
        for n, name in enumerate(ids):
            img_dir = os.path.join(data_dir, name, 'Images')
            img_file = os.path.join(img_dir, os.listdir(img_dir)[0])
            img = cv2.imread(img_file)
            img = resize(img, (shape[0], shape[1]), mode='constant', preserve_range=True)
            x[n] = img

            mask = np.zeros((shape[0], shape[1], 1), dtype=bool)
            mask_dir = os.path.join(data_dir, name, 'Masks')
            mask_file = os.path.join(mask_dir, os.listdir(mask_dir)[0])
            mask_read = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask_read = np.expand_dims(resize(mask_read, (shape[0], shape[1]), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_read)
            y[n] = mask
        return x, y

    def plot(self, history, path):
        fig, axs = plt.subplots(3)
        fig.set_figheight(20)
        fig.set_figwidth(10)
        x = range(1, len(history.history['loss'])+1)
        
        axs[0].plot(x, history.history['loss'], marker='o', markersize=ms, color='mediumvioletred', linewidth=0.5)
        #axs[0].plot(x, history.history['val_loss'], marker='o', markersize=ms, color='mediumpurple', linewidth=0.5)
        axs[0].set_title('Losses')
        axs[0].set_ylim((0, 1))
        #axs[0].legend(['train', 'val'], loc='upper right')

        axs[1].plot(x, history.history['accuracy'], marker='o', markersize=ms, color='mediumvioletred', linewidth=0.5)
        #axs[1].plot(x, history.history['val_accuracy'], marker='o', markersize=ms, color='mediumpurple', linewidth=0.5)
        axs[1].set_title('Accuracies')

        axs[2].plot(x, history.history['dice_coef'], marker='o', markersize=ms, color='mediumvioletred', linewidth=0.5)
        #axs[2].plot(x, history.history['val_dice_coef'], marker='o', markersize=ms, color='mediumpurple', linewidth=0.5)
        axs[2].set_title('Dice coef plots')

        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, 'loss_acc_dice.png'))

        plt.show()

    # uses Albumentations library
    def alb_aug(self, image, mask):
        mask = np.float32(mask) # img already converted
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5), #default (-0.2, 0.2) for contrast and brightness
            A.Affine(scale=[0.75, 1], shear=[-30, 30], p=0.5)
        ])
        transformed = transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    # no EarlyStop
    # no val
    def train_single(self):
        aug_x_train = np.zeros((self.x_train.shape[0], 128, 128, 3))
        aug_y_train = np.zeros((self.y_train.shape[0], 128, 128, 1))
        for ind in range(self.x_train.shape[0]):
            aug_x_train[ind], aug_y_train[ind] = self.alb_aug(self.x_train[ind], self.y_train[ind])
        seg_model = self.model(
            input_size=(128, 128, 3),
            filter_num=self.filter_num,
            n_labels=1,
            stack_num_down=2,
            stack_num_up=2,
            activation='ReLU',
            output_activation='Sigmoid',
            name='seg_model',
            backbone=self.backbone,
            weights=self.weights
        )
        seg_model.compile(
            loss='binary_crossentropy', 
            optimizer=Adam(learning_rate = self.lr), 
            metrics=['accuracy', losses.dice_coef]
        )
        history = seg_model.fit(
            aug_x_train, aug_y_train, 
            verbose=1,
            batch_size=batch_size, 
            shuffle=False,
            epochs=self.epochs,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(
                    lr_scheduler, verbose=1
                )
            ]
        )
        self.plot(history, os.path.join(self.test_save_dir))
        seg_model.save(os.path.join(self.test_save_dir, 'model'), overwrite=True)
        loss, acc, dice = seg_model.evaluate(
            self.x_test, self.y_test, verbose=1
        )
        with open(os.path.join(self.test_save_dir, 'test_scores.txt'), 'w') as text_file:
            text_file.write(
                'Test loss >>> ' + str(loss) + '\n\n'
                + 'Test acc >>> ' + str(acc) + '\n\n'
                + 'Test dice >>> ' + str(dice) + '\n\n'
            )        

    def train(self):
        histories = []
        val_losses = []
        val_accs = []
        val_dices = []
        kfold = KFold(n_splits=K, shuffle=True)
        for i, (train, val) in enumerate(kfold.split(self.x_train, self.y_train)):
            model_path = os.path.join(self.save_path, 'model'+str(i))
            print("Beginning training on fold {0}".format(i))
            print('Train: {0}'.format(train.shape))
            print('Val: {0}'.format(val.shape))
            aug_x_train = np.zeros((self.x_train.shape[0], 128, 128, 3))
            aug_y_train = np.zeros((self.y_train.shape[0], 128, 128, 1))
            for ind in range(self.x_train.shape[0]):
                aug_x_train[ind], aug_y_train[ind] = self.alb_aug(self.x_train[ind], self.y_train[ind])
            seg_model = self.model(
                input_size=(128, 128, 3),
                filter_num=self.filter_num,
                n_labels=1,
                stack_num_down=2,
                stack_num_up=2,
                activation='ReLU',
                output_activation='Sigmoid',
                name='seg_model',
                backbone=self.backbone,
                weights=self.weights
            )
            seg_model.compile(
                loss='binary_crossentropy', 
                optimizer=Adam(learning_rate = self.lr), 
                metrics=['accuracy', losses.dice_coef]
            )
            history = seg_model.fit(
                aug_x_train[train], aug_y_train[train], 
                verbose=1,
                batch_size=batch_size,
                validation_data=(
                    self.x_train[val], 
                    self.y_train[val]), 
                shuffle=False,
                epochs=self.epochs,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=model_path,
                        monitor='val_dice_coef',
                        verbose=1,
                        save_best_only=True,
                        save_freq='epoch', # required for custom metrics, periods=20 supplies true frequency
                        mode='max',
                        period=20
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_dice_coef',
                        verbose=1,
                        patience=40,
                        mode='max'
                    ),
                    tf.keras.callbacks.LearningRateScheduler(
                        lr_scheduler, verbose=1
                    )
                ]
            )
            print('Metrics names:', seg_model.metrics_names)
            self.plot(history, os.path.join(self.save_path, 'figs', 'fold{0}'.format(i)))
            print('Retraining model from', model_path)
            saved_model = tf.keras.models.load_model(
                os.path.join(model_path),
                custom_objects = {'dice_coef': losses.dice_coef}
            )
            val_loss, val_acc, val_dice = saved_model.evaluate(self.x_train[val], self.y_train[val])
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_dices.append(val_dice)
            print('Val loss, val acc, val dice resp:', val_loss, val_acc, val_dice)
            shutil.rmtree(model_path)
        with open(os.path.join(self.save_path, 'cv_scores.txt'), 'a') as text_file:
            text_file.write(
                'Val losses >>> ' + str(val_losses) + '\n\n'
                + 'Val accs >>> ' + str(val_accs) + '\n\n'
                + 'Val dices >>> ' + str(val_dices) + '\n\n'
                + 'Val dice avg: ' + str(sum(val_dices)/len(val_dices))
            )

if __name__ == '__main__':
    print('Running seg_pipline.py...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "specs.txt"), "w") as text_file:
        text_file.write(
            'Backbone: '+sys.argv[1]+'\n'
           +'Weights: '+sys.argv[2]+'\n'
           +'Model: '+sys.argv[3]+'\n'
           +'LR: '+str(sys.argv[6])+'\n'
           +'Epochs: '+str(sys.argv[7])+'\n'
           +'Filters: '+str(sys.argv[8])
        )

    seg_pipeline = SegPipeline(
        *sys.argv[1:6],
        sys.argv[6],
        float(sys.argv[7]),
        int(sys.argv[8]),
        int(sys.argv[9])
    )
    
    seg_pipeline.train()

    # for testing
    # seg_pipeline.train_single()










