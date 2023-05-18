#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ������ؿ�
import cv2
import numpy as np
import os
import joblib
import pandas as pd
import os   # ���ڷ��ʲ���ϵͳ����
import random   # ���ڶ����ݼ����������
import cv2   # ���ڶ�ȡ�ʹ���ͼ��
import numpy as np   # �������ݴ���
from sklearn.tree import DecisionTreeClassifier   # ������������
from sklearn.metrics import accuracy_score  # ������������������
import joblib   # ����ģ�͵ı���ͼ���
import pickle   # ���� pickle ���л��ͷ����л�

# �������ļ���
CLF_FILENAME = 'classifier.joblib'


# ��ȡ������˹������ͼ������
def get_pyramid_laplacian_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pyramid_level = 3   # ����������
    win_size = 3   # ������˹���Ӵ�С
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=win_size)
    pyramid = [laplacian]
    for i in range(pyramid_level - 1):
        laplacian = cv2.pyrDown(laplacian)   # �������²���
        pyramid.append(laplacian)
    features = np.concatenate([p.reshape(-1) for p in pyramid])
    return features


# ��ȡ��Եֱ��ͼ����
def get_edge_histogram_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed = cv2.GaussianBlur(gray, (3, 3), 0)   # ��˹ƽ������
    preprocessed = cv2.Canny(preprocessed, 50, 150)   # Canny ��Ե���
    hist_size = 5   # ֱ��ͼ��С
    hist = cv2.calcHist([preprocessed], [0], None, [hist_size], [0, 360])   # ����ֱ��ͼ
    hist = cv2.normalize(hist, hist).flatten()   # ��ֱ��ͼ��һ������ת��Ϊһά����
    return hist


# ��ȡ��ɫֱ��ͼ����
def get_color_histogram_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_size = [16, 8, 8]   # ֱ��ͼ��С
    ranges = [0, 180, 0, 256, 0, 256]   # ֱ��ͼ��Χ
    hist = cv2.calcHist([hsv], [0, 1, 2], None, hist_size, ranges)   # ����ֱ��ͼ
    hist = cv2.normalize(hist, hist).flatten()   # ��ֱ��ͼ��һ������ת��Ϊһά����
    return hist


# �������ݼ�
def create_dataset(img_list):
    dir_path = 'E:/For_new_PC/Tuning_tool/aituning/target/'   # ͼ��Ŀ¼
    X = []
    y = []
    for img_name in img_list:
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        laplacian_feats = get_pyramid_laplacian_features(img)
        edge_hist_feats = get_edge_histogram_features(img)
        color_hist_feats = get_color_histogram_features(img)
        features = np.concatenate([laplacian_feats, edge_hist_feats, color_hist_feats])
        X.append(features)
        y.append(get_labels_from_filename(img_name))   # ��ȡͼ���ǩ
    return np.array(X), np.array(y)


# ��ȡͼ���ǩ
def get_labels_from_filename(filename):
    # ����򵥵�ʵ����һ�����������������ǩ�����ѡȡһ��
    labels = ['cat', 'dog']
    label = random.choice(labels)
    return label

# ѵ��������
def train_classifier(X, y):
    decision_tree = DecisionTreeClassifier(max_depth=3)   # ������������
    decision_tree.fit(X, y)
    return decision_tree


# ���������
def save_classifier(clf, filename):
    joblib.dump(clf, filename, compress=3)


def load_classifier(filename):
    return joblib.load(filename)


def load_dataset():
    dir_path = 'E:/For_new_PC/Tuning_tool/aituning/target/'   # ͼ��Ŀ¼
    img_list = os.listdir(dir_path)
    img_list = [img_name for img_name in img_list if img_name.endswith('.jpg')]
    X, y = create_dataset(img_list)
    return X, y


def partition_dataset(X, y, num_folds=5):
    data = list(zip(X, y))
    random.shuffle(data)
    fold_size = len(X) // num_folds
    partitions = []
    for i in range(num_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size
        partition_X = []
        partition_y = []
        for j in range(start_idx, end_idx):
            partition_X.append(data[j][0])
            partition_y.append(data[j][1])
        partitions.append((partition_X, partition_y))
    return partitions


def evaluate_classifier(X, y, clf, num_folds=5):
    partitions = partition_dataset(X, y, num_folds=num_folds)
    accuracies = []
    for i in range(num_folds):
        test_X, test_y = partitions[i]
        train_X = []
        train_y = []
        for j in range(num_folds):
            if j != i:
                if partitions[j][0]:
                    train_X += partitions[j][0]
                    train_y += partitions[j][1]
        if not train_X:
            continue
        clf.fit(train_X, train_y)
        train_X = np.array(train_X)
        test_X = np.array(test_X)
        train_X = train_X.reshape((len(train_X), -1))
        test_X = test_X.reshape((len(test_X), -1))
        pred_y = clf.predict(test_X)
        acc = accuracy_score(test_y, pred_y)
        accuracies.append(acc)
    return np.mean(accuracies)


def main():
    # Load dataset
    X, y = load_dataset()

    # Evaluate classifier with cross-validation
    accuracies = []
    num_folds = 5
    partitions = partition_dataset(X, y, num_folds=num_folds)
    for i in range(num_folds):
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        for j in range(num_folds):
            if j == i:
                test_X += partitions[j][0]
                test_y += partitions[j][1]
            else:
                train_X += partitions[j][0]
                train_y += partitions[j][1]

        # Train classifier and test it
        clf = train_classifier(train_X, train_y)
        train_X = np.array(train_X)
        test_X = np.array(test_X)
        if train_X.size == 0:
            continue
        train_X = train_X.reshape((len(train_X), -1))
        test_X = test_X.reshape((len(test_X), -1))
        pred_y = clf.predict(test_X)
        acc = accuracy_score(test_y, pred_y)
        accuracies.append(acc)

    # Output cross-validation accuracy
    print('Cross-validation accuracy: {:.2f}'.format(np.mean(accuracies)))

    # Train the full classifier and save it
    clf = train_classifier(X, y)
    save_classifier(clf, CLF_FILENAME)
    print('Classifier saved as {}'.format(CLF_FILENAME))

    # Save/load the model using pickle instead of joblib
    model_path = os.path.join(os.getcwd(), 'E:/For_new_PC/Tuning_tool/aituning/pkl_output/', 'model.pkl')  # Change to your custom path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print('Classifier saved as {}'.format(model_path))

    # Predict the label of multiple images using the loaded classifier
    img_dir = 'E:/For_new_PC/Tuning_tool/aituning/target/'
    for img_path in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_path)
        img = cv2.imread(img_path)
        if img is None or img.size == 0:  # Check if the image was read successfully
            print('Error: could not read image {}'.format(img_path))
            continue
        X_test = np.array([np.concatenate([get_pyramid_laplacian_features(img), get_edge_histogram_features(img),
                                            get_color_histogram_features(img)])])
        pred_label = clf.predict(X_test)[0]
        print('Predicted label for {}: {}'.format(img_path, pred_label))

if __name__ == '__main__':
    main()

def extract_features(img):
    # Get the Laplacian, edge histogram, and color histogram features of the image
    laplacian_features = get_pyramid_laplacian_features(img)
    edge_hist_features = get_edge_histogram_features(img)
    color_hist_features = get_color_histogram_features(img)
    if laplacian_features is None or edge_hist_features is None or color_hist_features is None:
        return None

    # Concatenate the three feature arrays into a one-dimensional NumPy array
    features = np.concatenate((laplacian_features.reshape(-1), edge_hist_features.reshape(-1), color_hist_features.reshape(-1)))
    assert features.ndim == 1, "Features should be one-dimensional array"
    assert features.shape == (6614517,), "Features shape does not match expected shape"
    return features.tolist()

def get_pyramid_laplacian_features(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build the Gaussian pyramid
    pyramid = [gray]
    for i in range(2):
        pyramid.append(cv2.pyrDown(pyramid[i]))
    print(f"Pyramid sizes: {[p.shape for p in pyramid]}")

    # Build the Laplacian pyramid
    laplacian_pyramid = []
    for i in range(2):
        shape = pyramid[i].shape
        dst_shape = ((shape[1]*2)-1, (shape[0]*2)-1) # New size: 2X-1
        up = cv2.pyrUp(pyramid[i+1], dstsize=dst_shape)
        laplacian = cv2.subtract(pyramid[i], up[:shape[0], :shape[1]])
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(pyramid[-1])
    print(f"Laplacian pyramid sizes: {[lp.shape for lp in laplacian_pyramid]}")

    # Concatenate the Laplacian pyramid levels into a single feature vector
    features = np.concatenate([lp.ravel() for lp in laplacian_pyramid])
    return features.reshape(1, -1)


def get_edge_histogram_features(img):
    # ��ȡͼ��ı�Եֱ��ͼ����
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed = cv2.GaussianBlur(gray, (3, 3), 0)
    preprocessed = cv2.Canny(preprocessed, 50, 150)
    hist_size = 5
    hist = cv2.calcHist([preprocessed], [0], None, [hist_size], [0, 360])
    hist = cv2.normalize(hist, hist).flatten()
    # ������ת��Ϊ��ȷ����״������
    return hist.reshape(1, -1)


def get_color_histogram_features(img):
    # ��ȡͼ�����ɫֱ��ͼ����
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_size = [16, 8, 8]
    ranges = [0, 180, 0, 256, 0, 256]
    hist = cv2.calcHist([hsv], [0, 1, 2], None, hist_size, ranges)
    hist = cv2.normalize(hist, hist).flatten()
    # ������ת��Ϊ��ȷ����״������
    return hist.reshape(1, -1)


def get_recommendations(results):
    # ���ݴ����б����ɽ���
    recommendations = []
    for idx, row in results.iterrows():
        if row['prediction'] != row['gt_label']:
            if row['importance'] == 'essential':
                if row['action'] == 'adjust':
                    recommendations.append(
                        f'Please adjust the parameter "{row["parameter"]}" to improve the performance.')
            else:
                if row['action'] == 'adjust':
                    recommendations.append(
                        f'You may try adjusting the parameter "{row["parameter"]}" to improve the performance.')
                elif row['action'] == 'check':
                    recommendations.append(
                        f'Please check the parameter "{row["parameter"]}" and try again.')
    return recommendations


model_path = os.path.join(os.getcwd(), 'E:/For_new_PC/Tuning_tool/aituning/pkl_output/', 'model.pkl')
classifier = joblib.load(model_path)
dir_path = 'E:/For_new_PC/Tuning_tool/aituning/images/'
img_list = os.listdir(dir_path)

results = pd.DataFrame(columns=['image', 'prediction', 'gt_label', 'parameter', 'importance', 'action'])
count = 0
for img_name in img_list:
    if not img_name.endswith('.jpg'):
        continue
    img_path = os.path.join(dir_path, img_name)
    img = cv2.imread(img_path)
    img_features = extract_features(img)
    # ����ȡ����������Ԥ��
    prediction = classifier.predict(img_features.reshape(1, -1))
    gt_label = img_name.split('_')[0]
    if prediction[0] != gt_label:
        print(f"[Error] {img_name} is wrongly classified as {prediction[0]}, ground truth label is {gt_label}.")
        if gt_label == 'max':
            results.loc[count] = [img_name, prediction[0], gt_label, 'exposure', 'essential', 'adjust']
        elif gt_label == 'min':
            results.loc[count] = [img_name, prediction[0], gt_label, 'exposure', 'essential', 'check']
        count += 1
    else:
        if gt_label == 'max':
            results.loc[count] = [img_name, prediction[0], gt_label, 'exposure', 'useful', 'adjust']
        elif gt_label == 'min':
            results.loc[count] = [img_name, prediction[0], gt_label, 'exposure', 'useful', 'check']
        count += 1

recommendations = get_recommendations(results)
for rec in recommendations:
    print(rec)
