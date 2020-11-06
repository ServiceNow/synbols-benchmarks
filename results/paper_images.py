import numpy as np
import logging
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO)

from synbols.data_io import pack_dataset
from synbols.drawing import Camouflage, color_sampler, Gradient, ImagePattern, NoPattern, SolidColor
from synbols.generate import generate_char_grid, dataset_generator, basic_attribute_sampler, add_occlusion, \
    flatten_mask_except_first
from synbols.fonts import LANGUAGE_MAP
from synbols.generate import rand_seed
from synbols.visualization import plot_dataset
import matplotlib.pyplot as plt

font_list = """\
jotione
lovedbytheking
flavors
mrbedfort
butterflykids
newrocker
smokum
jimnightshade
""".splitlines()


def make_image(attr_sampler, file_name):
    x, _, y = pack_dataset(dataset_generator(attr_sampler, 1000))

    plot_dataset(x, y, h_axis='font', v_axis='char')

    plt.savefig(file_name)


def savefig(file_name):
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)


def show_fonts(seed):
    rng = np.random.RandomState(seed)

    def attr_sampler():
        for char in 'abCD':
            for font in font_list:
                yield basic_attribute_sampler(
                    alphabet=LANGUAGE_MAP['english'], char=char, font=font, is_bold=False, is_slant=False,
                    resolution=(128, 128), pixel_noise_scale=0)(seed=rand_seed(rng))

    x, _, y = pack_dataset(dataset_generator(attr_sampler(), 1000))
    plot_dataset(x, y, h_axis='font', v_axis='char', hide_axis=True)

    # savefig('fonts.png')


def show_languages(seed):
    language_list = ['korean',
                     'chinese',
                     'telugu',
                     'thai',
                     'gujarati',
                     'arabic',
                     'tamil',
                     'russian']

    rng = np.random.RandomState(seed)

    def attr_sampler():
        for lang in language_list:
            alphabet = LANGUAGE_MAP[lang].get_alphabet()
            for i in range(4):
                yield basic_attribute_sampler(
                    alphabet=alphabet, char=lambda rng: rng.choice(alphabet.symbols),
                    font=lambda rng: rng.choice(alphabet.fonts),
                    is_bold=False, is_slant=False, resolution=(128, 128), pixel_noise_scale=0)(seed=rand_seed(rng))

    x, _, y = pack_dataset(dataset_generator(attr_sampler(), 1000))
    h_values, v_values = plot_dataset(x, y, h_axis='alphabet', v_axis=None, n_col=len(language_list), n_row=4,
                                      hide_axis=True)

    # map = {'chinese-simplified': 'chinese'}
    # h_values = [map.get(val, val) for val in h_values]

    ax = plt.gca()
    ax.set_xticks((np.arange(len(h_values)) + 0.5) * x.shape[1])
    ax.set_xticklabels(h_values, rotation=0)
    ax.get_xaxis().set_visible(True)
    plt.xlabel('')

    # savefig('language.png')


def show_background(seed):
    rng = np.random.RandomState(seed)
    kwargs = dict(resolution=(128, 128), alphabet=LANGUAGE_MAP['english'].get_alphabet(), char='a', inverse_color=False,
                  pixel_noise_scale=0)
    attr_list = [
        basic_attribute_sampler(background=SolidColor((0.2, 0.2, 0)), foreground=SolidColor((0.8, 0, 0.8)), **kwargs),
        basic_attribute_sampler(background=lambda _rng: Gradient(types=('radial',), seed=rand_seed(_rng)),
                                foreground=lambda _rng: Gradient(types=('radial',), seed=rand_seed(_rng)),
                                **kwargs),
        basic_attribute_sampler(background=lambda _rng: Camouflage(stroke_angle=np.pi / 4, seed=rand_seed(_rng)),
                                foreground=lambda _rng: Camouflage(stroke_angle=np.pi * 3 / 4, seed=rand_seed(_rng)),
                                **kwargs),
        basic_attribute_sampler(background=lambda _rng: ImagePattern(seed=rand_seed(_rng)),
                                foreground=lambda _rng: ImagePattern(seed=rand_seed(_rng)),
                                **kwargs),
        add_occlusion(basic_attribute_sampler(**kwargs), n_occlusion=3,
                      scale=lambda _rng: 0.3 * np.exp(_rng.randn() * 0.1),
                      translation=lambda _rng: tuple(_rng.rand(2) * 2 - 1))
    ]

    def attr_sampler():
        for attr in attr_list:
            yield attr(seed=rand_seed(rng))

    x, _, y = pack_dataset(dataset_generator(attr_sampler(), 1000, flatten_mask_except_first))
    plot_dataset(x, y, h_axis='scale', v_axis=None, n_col=5, n_row=1, hide_axis=True)

    ax = plt.gca()
    ax.set_xticks((np.arange(5) + 0.5) * x.shape[1])
    ax.set_xticklabels(['Solid', 'Gradient', 'Camouflage', 'Natural', 'Occlusions'], rotation=0)
    ax.get_xaxis().set_visible(True)
    plt.xlabel('')

    # savefig('background.png')


def pack_dataset_resample(generator, resolution=128):
    """Turn a the output of a generator of (x,y) pairs into a numpy array containing the full dataset"""
    x, mask, y = zip(*generator)
    x = [zoom(img, (resolution / img.shape[0],) * 2 + (1,), order=0) for img in x]
    return np.stack(x), y


def show_resolution(seed):
    rng = np.random.RandomState(seed)
    kwargs = dict(alphabet=LANGUAGE_MAP['english'].get_alphabet(), is_bold=False, is_slant=False,
                  inverse_color=False, pixel_noise_scale=0)
    attr_list = [
        basic_attribute_sampler(resolution=(8, 8), char='b', font='arial', scale=0.9, rotation=0,
                                background=SolidColor((0, 0, 0)),
                                foreground=SolidColor((0.5, 0.5, 0)), **kwargs),
        basic_attribute_sampler(resolution=(16, 16), char='x', font='time', scale=0.7, **kwargs),
        basic_attribute_sampler(resolution=(32, 32), char='g', font='flavors', scale=0.6, rotation=1, **kwargs),
        basic_attribute_sampler(resolution=(64, 64), scale=0.3, n_symbols=5, **kwargs),
        basic_attribute_sampler(resolution=(128, 128), scale=0.1, n_symbols=30, **kwargs),
    ]

    def attr_sampler():
        for attr in attr_list:
            yield attr(seed=rand_seed(rng))

    x, y = pack_dataset_resample(dataset_generator(attr_sampler(), 1000))
    plot_dataset(x, y, h_axis='rotation', v_axis=None, n_col=5, n_row=1, hide_axis=True)

    ax = plt.gca()
    ax.set_xticks((np.arange(len(attr_list)) + 0.5) * x.shape[1])
    ax.set_xticklabels(['8 x 8', '16 x 16', '32 x 32', '64 x 64', '128 x 128'], rotation=0)
    ax.get_xaxis().set_visible(True)
    plt.xlabel('')

    # savefig('resolution.png')


def alphabet_sizes():
    for name, alphabet in LANGUAGE_MAP.items():
        print(name, len(alphabet.symbols))


if __name__ == "__main__":
    # plt.figure('languages', figsize=(5, 3))
    # show_languages()
    #
    # plt.figure('fonts', figsize=(5, 3))
    # show_fonts()
    #
    # plt.figure('resolution', figsize=(5, 3))
    # show_resolution()
    #
    # plt.figure('background', figsize=(5, 3))
    # show_background()

    # alphabet_sizes()

    for i in range(1):
        plt.figure('group %d' % i, figsize=(10, 6))

        plt.subplot(2, 2, 1)
        show_fonts(6)
        plt.title('a) fonts')

        plt.subplot(2, 2, 2)
        show_languages(3)
        plt.title('b) languages')

        plt.subplot(2, 2, 3)
        show_resolution(1)
        plt.title('c) resolution')

        plt.subplot(2, 2, 4)
        show_background(2)
        plt.title('d) background and foreground')

        savefig('group %d.png' % i)
        # plt.show()
