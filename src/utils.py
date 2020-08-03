import matplotlib.pyplot as plt
import numpy as np


def plot_image(image):
    image = image[:,:,::-1] if len(image.shape) == 3 else image
    plt.imshow(image); plt.show()


def plot_images_list(images_list):

    num_rows = len(images_list)
    num_cols = len(images_list[0])

    fig, axes = plt.subplots(num_rows, num_cols)
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 1)

    for i, image_list in enumerate(images_list):
        for j, image in enumerate(image_list):
            ax = axes[i, j]
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if len(image.shape) == 3:
                ax.imshow(image[:, :, ::-1])
            else:
                ax.imshow(image[:, :], cmap="gray")
    plt.show()