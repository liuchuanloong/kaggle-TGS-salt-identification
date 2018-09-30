import matplotlib.pyplot as plt
import numpy as np

def show_image_mask(im, mask, n=3, label='Image', show=True, cmap='jet', format='channels_first'):
    im = np.squeeze(im)
    mask = np.squeeze(mask)
    if format == 'channels_first':
        im = im.transpose((0, 2, 3, 1))

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(2, n)
    for i in range(n):
        axs[0, i].imshow(im[idx[i], :, :, :], cmap=cmap)
        axs[0, i].set_title('{}: {}'.format(label, i))
        axs[1, i].imshow(mask[idx[i], :, :], cmap=cmap)
        axs[1, i].set_title('mask: {}'.format(i))
    if show:
        plt.show()

def show_image_mask_pred(im, mask, logit,  n=3, label='Image', show=True, cmap='gray', format='channels_first'):
    mask = np.squeeze(mask)
    logit = np.squeeze(logit)

    if format == 'channels_first':
        if im.shape[1] == 1:
            im = np.squeeze(im)
            im_cmap = 'gray'
        else:
            im = im.transpose((0, 2, 3, 1))
            im_cmap = None

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(3, n)
    for i in range(n):
        axs[0, i].imshow(im[idx[i]], cmap=im_cmap)
        axs[0, i].set_title('{}: {}'.format(label, i))
        axs[1, i].imshow(mask[idx[i]], cmap=cmap)
        axs[1, i].set_title('mask: {}'.format(i))
        axs[2, i].imshow(logit[idx[i]], cmap=cmap)
        axs[2, i].set_title('logit: {}'.format(i))
    if show:
        plt.show

def show_image_tta_pred(im, tta_im, logit, tta_logit,  n=3, label='Image', show=True, cmap='gray', format='channels_first'):
    logit = np.squeeze(logit)
    tta_logit = np.squeeze(tta_logit)

    if format == 'channels_first':
        if im.shape[1] == 1:
            im = np.squeeze(im)
            tta_im = np.squeeze(tta_im)
            im_cmap = 'gray'
        else:
            im = im.transpose((0, 2, 3, 1))
            tta_im = tta_im.transpose((0, 2, 3, 1))
            im_cmap = None

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(n, 4)
    for i in range(n):
        axs[i, 0].imshow(im[idx[i]], cmap=im_cmap)
        axs[i, 0].set_title('{}: {}'.format(label, i))
        axs[i, 1].imshow(logit[idx[i]], cmap=cmap)
        axs[i, 1].set_title('logit: {}'.format(i))
        axs[i, 2].imshow(tta_im[idx[i]], cmap=im_cmap)
        axs[i, 2].set_title('TTA {}: {}'.format(label, i))
        axs[i, 3].imshow(tta_logit[idx[i]], cmap=cmap)
        axs[i, 3].set_title('TTA logit: {}'.format(i))
    if show:
        plt.show()