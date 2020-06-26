import os
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

N = 10
KLIST = [1, 5, 10, 20, 30, 50, 100, 150, 200]
K = 10
SIZE = (48, 48)
# SIZE = (24, 24)
LEN = SIZE[0] * SIZE[1]


def img2vector(img):
    v = np.reshape(img, (LEN, 1))
    return v


def vector2img(v):
    v_ = np.reshape(v, SIZE)
    img = Image.fromarray(v_).convert('RGB')
    return img


def get_data(file_path):
    data_ = {}
    visual = "./visual/"
    for root, dir, file_list in os.walk(file_path):
        # print(root)
        for file in file_list:
            file_split = file.split('.')
            if file_split[-1] == 'pgm':
                im = Image.open("./Recognition/" + file)
                im.save(visual + file_split[0] + '.' + file_split[1] + '.jpg')
                img = np.array(im.resize(SIZE, Image.BILINEAR)).reshape(LEN)
                if file_split[0] in data_:
                    data_[file_split[0]].append(img)
                else:
                    data_.setdefault(file_split[0], [])
                    data_[file_split[0]].append(img)
                # print(img.shape)
                # print(file)
    print("reading done")

    # split data
    train_index = random.sample(range(11), N)
    test_index = list(set(range(11)) - set(train_index))
    print(train_index)
    print(test_index)
    train_data_ = []
    train_label_ = []
    test_data_ = []
    test_label_ = []
    for key, values in data_.items():
        for i in train_index:
            train_data_.append(values[i])
            train_label_.append(int(key[-2:]))
        for i in test_index:
            test_data_.append(values[i])
            test_label_.append(int(key[-2:]))
    print("train data: ", np.shape(train_data_))
    print("test data: ", np.shape(test_data_))
    print(train_label_)
    print(test_label_)
    print("spliting done")
    return np.array(train_data_), np.array(train_label_), np.array(test_data_), np.array(test_label_)


def face_recognition(im, eig_vectors_real, train_pattern):
    n_class = len(train_pattern)
    distance = np.zeros(n_class)
    y = np.dot(im.T, eig_vectors_real)
    # print("y: ", np.shape(y))
    for i in range(n_class):
        distance[i] = np.linalg.norm(y - train_pattern[i, :])
    im_class = np.argmin(distance) + 1
    return im_class


def eigenface(train_data_, train_label_, test_data_, test_label_):
    # calculate the mean face
    all_mean = np.mean(train_data_, axis=0)
    im_mean = Image.fromarray(all_mean.reshape(SIZE)).convert('RGB')
    im_mean.save("./eigenface/mean_face.jpg")
    print("mean done")

    # calculate the cov mat
    n_train = len(train_data_)
    n_class = int(n_train / N)
    print("training images: ", n_train)
    print("image classes: ", n_class)
    mat = np.zeros((LEN, LEN))
    for i in range(n_train):
        x = np.mat(train_data_[i, :] - all_mean)
        # print(np.shape(x))
        mat += np.dot(x.T, x)
    mat /= n_train

    # calculate the eigen values and eigen vectors
    eig_values, eig_vectors = np.linalg.eig(mat)
    eig_values = np.real(eig_values)
    eig_vectors = np.real(eig_vectors)
    face_feature = np.dot(all_mean, eig_vectors)
    np.savetxt('./eigenface/eig_vectors.txt', eig_vectors)
    np.savetxt('./eigenface/face_feature.txt', face_feature)
    print(np.shape(eig_values))
    print(np.shape(eig_vectors))

    for i in range(20):
        v = np.reshape(eig_vectors[:, i], SIZE)
        # img = Image.fromarray(v).convert('RGB')
        # img = Image.fromarray(np.ones_like(v) * 128 + v / np.max(abs(v)) * 128).convert('RGB')
        img = Image.fromarray(v / np.max(abs(v)) * 255).convert('RGB')
        img.save('./eigenface/feture_face{0}.jpg'.format(i+1))

    # use the first k eigen vectors
    n_test = len(test_data_)
    print("testing images: ", n_test)
    acc_ = []
    for k in KLIST:
        eig_vectors_real = np.real(eig_vectors[:, :k])
        # eig_values_real = np.real(eig_values[:k])
        # print("k vec shape: ", np.shape(eig_vectors_real))

        # save data of each class
        train_pattern = np.zeros((n_class, k))
        for i in range(n_class):
            tr = np.mean(train_data_[i * N:(i + 1) * N, :], axis=0)
            # print("train chip", np.shape(tr))
            train_pattern[i, :] = np.dot(tr, eig_vectors_real)

            t = train_pattern[i, :]
            im = all_mean.reshape(SIZE)
            for j in range(k):
                v = np.reshape(eig_vectors_real[:, j], SIZE)
                im = np.add(v * t[j], im)
            img = Image.fromarray(im).convert('RGB')
            img.save('./eigenface/pattern_face{}_{}_{}.jpg'.format(N, k, i + 1))

        ac = 0
        for i in range(n_test):
            im = test_data_[i, :]
            im_class = face_recognition(im, eig_vectors_real, train_pattern)
            # print(im_class, test_label_[i])
            if im_class == test_label_[i]:
                ac += 1
        ac = ac / n_test
        acc_.append(ac)
        # print('图片{0}分类结果{1}'.format(i+1, im_class))
    return acc_


def fisherface(train_data_, train_label_, test_data_, test_label_):
    all_mean = np.mean(train_data_, axis=0)
    n_train = len(train_data_)
    n_class = int(n_train / N)
    print("training images: ", n_train)
    print("image classes: ", n_class)
    class_mean = np.zeros(shape=(n_class, LEN))
    for i in range(n_class):
        class_mean[i, :] = np.mean(train_data_[i * N:(i + 1) * N, :], axis=0)
    print(np.shape(class_mean))

    sw = np.zeros((LEN, LEN))
    for i in range(n_train):
        x = np.mat(train_data_[i, :] - class_mean[train_label_[i] - 1, :])
        # print(np.shape(x))
        sw += np.dot(x.T, x)
    print("sw: ", np.shape(sw))

    sb = np.zeros((LEN, LEN))
    for i in range(n_class):
        x = class_mean[i, :] - all_mean
        sb += N * np.dot(x, x.T)
    print("sb: ", np.shape(sb))

    w = np.dot(np.linalg.inv(sw), sb)
    eig_values, eig_vectors = np.linalg.eig(w)
    eig_values = np.real(eig_values)
    eig_vectors = np.real(eig_vectors)
    face_feature = np.dot(all_mean, eig_vectors)
    np.savetxt('./fisherface/eig_vectors.txt', eig_vectors)
    np.savetxt('./fisherface/face_feature.txt', face_feature)
    print(np.shape(eig_values))
    print(np.shape(eig_vectors))

    n_test = len(test_data_)
    print("testing images: ", n_test)
    acc_ = []
    for k in KLIST:
        eig_vectors_real = np.real(eig_vectors[:, :k])
        # print("k vec shape: ", np.shape(eig_vectors_real))

        # save data of each class
        train_pattern = np.zeros((n_class, k))
        for i in range(n_class):
            tr = np.mean(train_data_[i * N:(i + 1) * N, :], axis=0)
            # print("train chip", np.shape(tr))
            train_pattern[i, :] = np.dot(tr, eig_vectors_real)

            # t = train_pattern[i, :]
            # im = x_mean.reshape(SIZE)
            # for j in range(k):
            #     v = np.reshape(eig_vectors_real[:, j], SIZE)
            #     im = np.add(v * t[j], im)
            # img = Image.fromarray(im).convert('RGB')
            # img.save('./fisherface/pattern_face{}_{}_{}.jpg'.format(N, k, i + 1))

        ac = 0
        for i in range(n_test):
            im = test_data_[i, :]
            im_class = face_recognition(im, eig_vectors_real, train_pattern)
            # print(im_class, test_label_[i])
            if im_class == test_label_[i]:
                ac += 1
        ac = ac / n_test
        acc_.append(ac)
        # print('图片{0}分类结果{1}'.format(i+1, im_class))
    return acc_


def detection(detection_path_, eig_vectors, face_feature):
    detect_data = []
    for root, dir, file_list in os.walk(detection_path_):
        # print(root)
        for file in file_list:
            file_split = file.split('.')
            if file_split[-1] == 'jpg':
                img = cv2.imread(detection_path + '/' + file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detect_data.append(img)
                # im = Image.open(detection_path + '/' + file).convert('L')
                # img = np.array(im)
                # print(np.shape(img))
    print("detecting... ")
    thresh = 6670
    box_size_list = [SIZE, SIZE*2, SIZE*4]
    box_size = box_size_list[0][0]
    box_size = 96
    # step = int(box_size / 2)
    step = 20

    img = detect_data[1]
    result = cv2.imread(detection_path + '/2.jpg')
    img_size = np.shape(img)
    print(np.shape(img))

    i = 0
    while i + box_size < img_size[0]:
        j = 0
        while j + box_size < img_size[1]:
            im = img[i:i+box_size, j:j+box_size]
            im = cv2.resize(im, SIZE, im)
            im = cv2.equalizeHist(im, im)
            # cv2.imshow("im", im)
            # cv2.waitKey()
            print(i, j, np.shape(im))
            im = im.reshape(1, LEN)
            y = np.dot(im, eig_vectors)
            distance = np.linalg.norm(y - face_feature)
            print(i, j, np.shape(im), distance)
            if distance < thresh:
                print("found at ", i, j)
                cv2.rectangle(result, (j, i), (j + box_size, i + box_size), (0, 0, 255))
            j += step
        i += step
    cv2.imwrite("./detectresults/result_slide_1.jpg", result)
    cv2.imshow("result", result)
    cv2.waitKey()
    print("detect done")

def detect2():
    img = cv2.imread('./Detection/3.jpg')
    result = cv2.imread('./Detection/3.jpg')
    cv2.imshow("image", img)  # 显示图像
    cv2.waitKey()

    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    faceRects = classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            print(x, y, w, h)
            # 框出人脸
            cv2.rectangle(result, (x, y), (x + h, y + w), (0, 0, 255), 2)
    cv2.imwrite("./detectresults/result_haar_3.jpg", result)
    # cv2.imshow("image", img)  # 显示图像
    # cv2.waitKey()
    return 0


if __name__ == "__main__":
    recognition_path = "./Recognition"
    train_data, train_label, test_data, test_label = get_data(recognition_path)
    ac = fisherface(train_data, train_label, test_data, test_label)
    print(ac)
    plt.plot(KLIST, ac, "-bo", label="try_just")
    plt.title('Accuracy in different PCs')
    plt.xlabel('Num of PCs')
    plt.ylabel('Accuracy')
    plt.savefig("./eigenface/accuracy_n{}.png".format(N))
    ac = eigenface(train_data, train_label, test_data, test_label)
    print(ac)
    plt.plot(KLIST, ac, "-r+", label="try_just")
    plt.title('Accuracy in different PCs(N={})'.format(N))
    plt.xlabel('Num of PCs')
    plt.ylabel('Accuracy')
    plt.savefig("./fisherface/accuracy_n{}.png".format(N))
    #
    # acc = np.zeros(len(KLIST))
    # m = 1
    # for i in range(m):
    #     ac = eigenface(train_data, train_label, test_data, test_label)
    #     ac = fisherface(train_data, train_label, test_data, test_label)
    #     acc += ac
    # acc /= m
    # print(acc)
    #
    # plt.plot(KLIST, acc, "-bo", label="try_just")
    # plt.title('Accuracy in different PCs')
    # plt.xlabel('Num of PCs')
    # plt.ylabel('Accuracy')
    # plt.savefig("./fisherface/accuracy_n{}.png".format(N))
    # plt.show()

    detection_path = "./Detection"
    eigenface_vectors = np.loadtxt('./eigenface/eig_vectors.txt')[:, :300]
    eigenface_feature = np.loadtxt('./fisherface/face_feature.txt')[:300]
    print(np.shape(eigenface_vectors))
    print(np.shape(eigenface_feature))
    # fisherface_vectors = np.loadtxt('./fisherface/eig_vectors.txt')[:, :300]
    # fisherface_feature = np.loadtxt('./fisherface/face_feature.txt')[:300]
    # print(np.shape(fisherface_vectors))
    # print(np.shape(fisherface_feature))
    detection(detection_path, eigenface_vectors, eigenface_feature)
    # detection(detection_path, fisherface_vectors, fisherface_feature)
    # print(detect2())
