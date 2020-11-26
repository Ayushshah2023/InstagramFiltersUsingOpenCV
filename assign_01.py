from scipy import ndimage, misc
import numpy as np
from skimage import filters, io, util
from math import pi


class Operators:
    def __init__(self, image, name):
        self.image = image
        self.image_name = name
    def roberts(self):
        roberts = filters.roberts(self.image)
        io.imsave(str('roberts_' + self.image_name), util.invert(roberts))
    # Implementation of Robinson operator (compass mask)
    def robinson(self, name):
        image = ndimage.imread(name)
        h0_3 = np.array([
            [[-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]],

            [[-2.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0]],

            [[-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0]],

            [[0.0, -1.0, -2.0],
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0]]
        ])

        h4_7 = np.negative(h0_3)
        e_k = np.zeros(image.shape)

        h0_7 = np.concatenate((h0_3, h4_7), axis=0)

        for filter in h0_7:
            e_k = np.maximum(ndimage.filters.convolve(image, filter), e_k)


        k_k = image
        for v in range(0, image.shape[1]):
            for u in range(0, image.shape[0]):
                k_k[u][v] = pi/4*e_k[u][v]

        misc.imsave(str('robinson_' + self.image_name), util.invert(k_k))


def main():

    images = ['lfc.jpg']
    for image_name in images:
        image = io.imread(image_name)
        print(image)
        operators = Operators(image, image_name)
        operators.roberts()
        operators.robinson(image_name)


if __name__ == '__main__':
    main()
