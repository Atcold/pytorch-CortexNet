import torch  # if torch is not imported BEFORE pyplot you get a FUCKING segmentation fault
from matplotlib import pyplot as plt
from os.path import isdir, join
from os import mkdir


def _hist_show(a, k):
    a = _to_view(a)
    plt.subplot(2, 3, k)
    plt.hist(a.reshape(-1), 50)
    plt.grid('on')
    plt.gca().axes.get_yaxis().set_visible(False)


def show_four(x, next_x, x_hat, fig):
    """
    Saves/overwrites a PDF named fig.pdf with x, next_x, x_hat histogram and x_hat

    :param x: x[t]
    :type x: torch.FloatTensor
    :param next_x: x[t + 1]
    :type next_x: torch.FloatTensor
    :param x_hat: ~x[t + 1]
    :type x_hat: torch.FloatTensor
    :param fig: figure number
    :type fig: int
    :return: nothing
    :rtype: None
    """
    f = plt.figure(fig)
    plt.clf()
    _sub(x, 1)
    _sub(next_x, 4)
    dif = next_x - x
    _sub(dif, 2)
    _hist_show(dif, 3)
    _sub(x_hat, 5)
    _hist_show(x_hat, 6)
    plt.subplots_adjust(left=0.01, bottom=0.06, right=.99, top=1, wspace=0, hspace=.12)
    f.savefig(str(fig) + '.pdf')


# Setup output folder for figures collection
def _show_ten_setup(pdf_path):
    if isdir(pdf_path):
        print('Folder "{}" already existent. Exiting.'.format(pdf_path))
        exit()
    mkdir(pdf_path)


def show_ten(x, x_hat, pdf_path='PDFs'):
    """
    First two rows 10 ~x[t + 1], second two rows 10 x[t]

    :param x: x[t]
    :type x: torch.FloatTensor
    :param x_hat: ~x[t + 1]
    :type x_hat: torch.FloatTensor
    :param pdf_path: saving path
    :type pdf_path: str
    :return: nothing
    :rtype: None
    """
    if show_ten.c == 0 and pdf_path: _show_ten_setup(pdf_path)
    if show_ten.c % 10 == 0: show_ten.f = plt.figure()
    plt.figure(show_ten.f.number)
    plt.subplot(4, 5, 1 + show_ten.c % 10)
    _img_show(x_hat, y0=-.16, s=8)
    plt.subplot(4, 5, 11 + show_ten.c % 10)
    _img_show(x, y0=-.16, s=8)
    show_ten.c += 1
    plt.subplots_adjust(left=0, bottom=0.02, right=1, top=1, wspace=0, hspace=.12)
    if show_ten.c % 10 == 0: show_ten.f.savefig(join(pdf_path, str(show_ten.c // 10) + '_10.pdf'))
show_ten.c = 0


def _img_show(a, y0=-.13, s=12):
    a = _to_view(a)
    plt.imshow(a)
    plt.title('<{:.2f}> [{:.2f}, {:.2f}]'.format(a.mean(), a.min(), a.max()), y=y0, fontsize=s)
    plt.axis('off')


def _sub(a, k):
    plt.subplot(2, 3, k)
    _img_show(a)


def _to_view(a):
    return a.cpu().numpy().transpose((1, 2, 0))


def _test_4():
    img = _test_setup()
    show_four(img, img, img, 1)


def _test_10():
    img = _test_setup()
    for i in range(20): show_ten(img, -img, '')


def _test_setup():
    from skimage.data import astronaut
    from skimage.transform import resize
    from matplotlib.figure import Figure
    Figure.savefig = lambda self, _: plt.show()  # patch Figure class to simply display the figure
    img = torch.from_numpy(resize(astronaut(), (256, 256)).astype('f4').transpose((2, 0, 1)))
    return img


if __name__ == '__main__':
    _test_4()
    _test_10()

__author__ = "Alfredo Canziani"
__credits__ = ["Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Development"  # "Prototype", "Development", or "Production"
__date__ = "Mar 17"
