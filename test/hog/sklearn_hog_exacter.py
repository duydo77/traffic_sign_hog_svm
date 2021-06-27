
from skimage.feature import hog
from skimage import exposure
import cv2
import matplotlib.pyplot as plt
# import imgutils

class skHogDescripter():
    pass


if __name__=='__main__':
    data_path = './image/00001.png'
    img = cv2.imread(data_path)
    
    (H, hogImage) = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                    visualize=True)
    # print(hogImage)
    print(hogImage.shape)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    
    hogImage = hogImage.astype("uint8")
    plt.imshow(hogImage)
    plt.show()