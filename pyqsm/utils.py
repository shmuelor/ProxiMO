import matplotlib.pyplot as plt


def save_screenshot(img, fn, coor=None):
    """Save a screenshot of a 3D image in axial, sagittal and coronal views.
    Parameters
    ----------
    img : numpy.ndarray
        3D image. Shape: (nx, ny, nz).
    fn : str
        File name.
    coor : list of 3 floats, optional
        Coordinate of the crosshair. Default: None.
    """
    shape = img.shape
    if coor == None:
        coor = [int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)]
    # axial
    img_ax = img[:, :, coor[2]]
    # sagittal
    img_sg = img[coor[0], :, :]
    # coronal
    img_cr = img[:, coor[1], :]
    # save
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img_ax, cmap="gray")
    plt.axis("off")
    plt.title("Axial")
    plt.subplot(2, 2, 2)
    plt.imshow(img_sg, cmap="gray")
    plt.axis("off")
    plt.title("Sagittal")
    plt.subplot(2, 2, 4)
    plt.imshow(img_cr, cmap="gray")
    plt.axis("off")
    plt.title("Coronal")
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
