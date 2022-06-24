import numpy as np
# from PIL import Image
#################################Mi codigo######################################
fname = 0

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def show_image_from_tensor(img,level):
    import matplotlib.pyplot as plt
    # print('img shape: {}'.format(img.shape))
    c, h, w = img.shape
    # h = w = 1
    img = img.detach().numpy()#.squeeze()
    img = np.transpose(img, (1,2,0))
    global fname
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w/100, h/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if c == 3:
        img = NormalizeData(img)
        ax.imshow(img, aspect='auto')
    else:
        ax.imshow(img, aspect='auto')
    # plt.show()
    fig.savefig('output/imgs/{:0>10d}_{}.jpg'.format(fname, level))
    plt.close()
    fname += 1