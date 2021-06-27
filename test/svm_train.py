import sklearn
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import glob
import os
import cv2
import argparse
import pandas as pd
import time
import numpy as np
import pickle
from sklearn import metrics

def data_feature_exacter(path, data_labels):
    print('[INFO] Load data and compute HOG of image from path ......')
    datas = []
    labels = []
    filenames = []

    for label in data_labels:
        image_paths = glob.glob(path + '/'+ label + '/*')
        for im_path in image_paths:
            im = cv2.imread(im_path)
            gray = cv2.resize(im, (96, 96))
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            hogFeature = hog(gray,orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            transform_sqrt=True,
                            visualize=False,
                            block_norm='L2')
            datas.append(hogFeature)
            labels.append(label)
            filenames.append(im_path)
    
    print('asdfadf ',len(datas))
    datas = np.stack(datas, axis = 0)
    print('data shape ', datas.shape)
    labels = np.stack(labels, axis = 0)
    return datas, labels, filenames

def load_test_data(path, data_labels):
    datas = []
    labels = []

    pf = pd.read_csv(path + '/test.csv')

    for row in tqdm(pf.iterrows()):
        l = str(row[1]['ClassId']) 
        if l in data_labels:
            im = cv2.imread(path + '/' + row[1].Path)
            gray = cv2.resize(im, (96, 96))
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            hogFeature = hog(gray,orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            transform_sqrt=True,
                            visualize=False,
                            block_norm='L2')
            datas.append(hogFeature)
            labels.append(l)
    return datas, labels 


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train_path", required=False, default='../GTSRB/train', help="Path to the training dataset")
    ap.add_argument("-v", "--val_path", required=False, default='../GTSRB/val', help="Path to the training dataset")
    args = vars(ap.parse_args())

    data_label = ('14', '17', '38', '39', '40') 
    '''
        14: stop
        17: cam
        38: di ben phai
        39: di ben trai
        40: vong xoay     
    '''
    # train_datas, train_labels, _ = data_feature_exacter(args["train_path"], data_label)
    val_datas, val_labels = load_test_data('../GTSRB', data_label)

    # svm = SGDClassifier(learning_rate='optimal', 
    #                     loss='hinge', penalty='l2', 
    #                     alpha=0.001, max_iter=15000, 
    #                     verbose=True, n_jobs=-1, 
    #                     tol=1e-3, early_stopping=True)

    # t = time.time()
    # svm.fit(train_datas, train_labels)
    # print ('[INFO] That took %fs' % (time.time() - t))
    
    # pickle.dump(svm, open('./model/model.dd', 'wb')) 
    

    model = pickle.load(open('./model/model.dd', 'rb'))
    
    im = cv2.imread('../GTSRB/Test/00003.png')
    gray = cv2.resize(im, (96, 96))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    hogFeature = hog(gray,orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            transform_sqrt=True,
                            visualize=False,
                            block_norm='L2')
    res = model.predict(val_datas)
    print(type(res))
    print('[INFO] Confusion matrix... \n', metrics.confusion_matrix(res, val_labels))