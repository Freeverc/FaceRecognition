import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

def vector2img(v):
    v_ = np.reshape(v, (112, 92))
    img = Image.fromarray(v_)
    return img

def refactor(eigVectors_real, x_mean, im, k):
    x = np.array(im).reshape((92*112, 1)) / 255
    x_ = x_mean.reshape((112, 92)) * 255
    for i in range(k):
        t = np.dot(eigVectors_real[:, i].T, np.add(x, -x_mean).reshape(112*92, 1))
        # print('t', np.shape(t))
        x_ = np.add(eigVectors_real[:, i].reshape((112, 92)) * t * 255, x_)
    im_ = Image.fromarray(x_)
    return im_


if __name__ == "__main__":
    file_name = input('输入图片文件名：')
    im = Image.open(file_name)
    x_mean = np.loadtxt('./model/face_mean.txt')
    im_mean = Image.fromarray(x_mean.reshape((112,92)) * 255).convert('RGB')
    # im_mean.show()
    im_mean.save('./results/平均脸.jpg')
    x_mean = x_mean.reshape((112 * 92, 1))
    eigVectors_real = np.loadtxt('./model/eigVectors_real.txt')
    for i in range(20):
        v = np.reshape(eigVectors_real[:,i], (112, 92))
        img = Image.fromarray(np.ones_like(v) * 128 + v / np.max(abs(v)) * 128).convert('RGB')
        img.save('./results/特征脸{0}.jpg'.format(i+1))

    kValues = [1, 5, 15, 35, 60, 100, 150, 200, 250, 300]
    for k in kValues:
        im_ = refactor(eigVectors_real, x_mean, im, k).convert('RGB')
        im_.save('./results/refact'+str(k)+'.jpg')
    im.convert('RGB').save('./results/原图.jpg')
    print('重构结果已经保存在results文件夹中')
    # plt.imshow(im)
    # plt.show()
    file_name = input('输入图片文件名：')
    d = input('输入能量百分比(1-100的整数)：')
    k = int(int(d) * 5)
    im = Image.open(file_name).convert('F')
    im_ = refactor(eigVectors_real, x_mean, im, k).convert('RGB')
    im_.save('./results/输入图片重构结果.jpg')
    im_.show()