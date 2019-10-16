import matplotlib.pyplot as plt

def savegrid(fig_path, images, predictions, \
    gauss_mu=None, labels=None, nrow=8, ncol=8, name='image'):
    step = 2
    ncol = 8
    fig_width = 20
    if labels is not None:
        step = 3
        ncol = 12
        fig_width = 30
    plt.rcParams['figure.figsize'] = (fig_width, 40)
    j = 0
    for i in range(0, nrow*ncol, step):
        if j >= len(images):
            break
        img = images[j].squeeze()
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img) #,interpolation='none', cmap="nipy_spectral")
        if gauss_mu is not None:
            for k in range(gauss_mu[j].shape[0]):
                y_jk = ((gauss_mu[j, k, 0]+1)*15*16).astype(np.int)
                x_jk = ((gauss_mu[j, k, 1]+1)*11*16).astype(np.int)
                plt.plot(x_jk, y_jk, 'bo')
        plt.title('{}_{}'.format(name, j))
        plt.axis('off')

        pred = predictions[j].squeeze()
        plt.subplot(nrow, ncol, i+2)
        plt.imshow(pred)
        plt.title('predict_{}'.format(j))
        plt.axis('off')

        if labels is not None:
            label = labels[j]
            plt.subplot(nrow, ncol, i+3)
            plt.imshow(label)
            plt.title('label_{}'.format(j))
            plt.axis('off')

        j += 1
    # plt.show()
    plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
    plt.close()
