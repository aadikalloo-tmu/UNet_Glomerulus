import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import math
import os

def train_test_split():
    np.random.seed(127)

    use = np.random.choice(np.array(range(0, len(imgs))), math.ceil(.8*len(imgs)), replace = False)
    imgs_arr = np.array(imgs)
    train = list(imgs_arr[use])
    test = list(imgs_arr[[i for i in range(0,len(imgs)) if i not in use]])
    masks_arr = np.array(masks)
    train_masks = list(masks_arr[use])
    test_masks = list(masks_arr[[i for i in range(0,len(masks_arr)) if i not in use]])

def main(densenet=True):
    masks = sorted(glob.glob('results/masks/*'))
    imgs = sorted(glob.glob('results/train/*'))
    train_X = []
    train_Y = []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256,256,1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            train_Y.append(mask)
            img = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
            img = img/255
            if densenet:
                img = -1*(img - np.max(img))
            img = img.reshape(256,256,1)
            train_X.append(img)

    masks = sorted(glob.glob('results/test_masks/*'))
    imgs = sorted(glob.glob('results/test/*'))
    test_X = []
    test_Y = []
    for i in range(len(masks)):
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(256,256,1)
        # only take patches which are a 1
        if np.sum(mask) > 0:
            test_Y.append(mask)
            img = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
            img = img/255
            if densenet:
                img = -1*(img - np.max(img))
            img = img.reshape(256,256,1)
            test_X.append(img)

    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.float32)
    test_X = np.array(test_X, dtype=np.float32)
    test_Y = np.array(test_Y, dtype=np.float32)

    print('train_X.shape', train_X.shape, 'test_X.shape', test_X.shape)
    plt.imshow(test_Y[14,...,0])
    plt.savefig('test_Y_0.png')
    plt.imshow(test_X[14,...])
    plt.savefig('test_Y.png')

    print('train_X', len(train_X))
    print('test_X', len(test_X))

if __name__ == '__main__':
    main()