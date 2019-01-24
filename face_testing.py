import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_test_data():
    test_data = []
    for i in range(1, 41):
        for j in range(6, 11):
            file_name = './data/' + 's' + str(i) + '/' + str(j) + '.pgm'
            # print(file_name)
            im = np.array(Image.open(file_name)).reshape((92 * 112, 1)) / 255
            test_data.append(im)
    return test_data

def face_classification(im, eigVectors_real, model, k):
    d = np.zeros((40))
    y = np.dot(im.T, eigVectors_real[:, :k])
    # print(np.shape(y))
    for i in range(40):
        d[i] += np.linalg.norm(y - model[i, :k])
    im_class = np.argmin(d) + 1
    return im_class

def model_testing(test_data, eigVectors_real, model, k):
    acc = 0
    for i in range(40):
        for j in range(5):
            im = test_data[i * 5 + j][:]
            im_class = face_classification(im, eigVectors_real, model, k)
            if im_class == i + 1:
                acc += 1
        # print('图片{0}分类结果{1}'.format(i+1, im_class))
    acc = 100 * acc / len(test_data)
    return acc

if __name__ == "__main__":
    eigVectors_real = np.loadtxt('./model/eigVectors_real.txt')
    model = np.loadtxt('./model/model_data.txt')

    test_data = get_test_data()
    print('测试集数量：', len(test_data))

    kValues = [1, 5, 15, 35, 60, 100, 150, 200, 250, 300]
    acc = np.zeros((len(kValues)))
    for i in range(len(kValues)):
        acc[i] = model_testing(test_data, eigVectors_real, model, kValues[i])
        print('主成分数量：{0},识别准确率：{1}%'.format(kValues[i], acc[[i]]))

    plt.plot(kValues, acc, "-bo", label="try_just")
    plt.title('Accuracy in different PCs')
    plt.xlabel('Num of PCs')
    plt.ylabel('Accuracy')
    plt.savefig("./results/accuracy.png")
    plt.show()
    k = 200
    file_name = input('输入文件名：')
    im = np.array(Image.open(file_name)).reshape((92 * 112, 1)) / 255
    im_class = face_classification(im, eigVectors_real, model, k)
    print('输入图片分类结果:{0}'.format(im_class))



