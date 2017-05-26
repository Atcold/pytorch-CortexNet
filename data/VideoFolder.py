import collections
import torch
import torch.utils.data as data

from random import shuffle as list_shuffle  # for shuffling list
from math import ceil
from os import listdir
from os.path import isdir, join, isfile
from itertools import islice
from numpy.core.multiarray import concatenate, ndarray
from skvideo.io import FFmpegReader, ffprobe
from torch.utils.data.sampler import Sampler
from torchvision import transforms as trn
from tqdm import tqdm
from time import sleep
from bisect import bisect

# Implement object from https://discuss.pytorch.org/t/loading-videos-from-folders-as-a-dataset-object/568

VIDEO_EXTENSIONS = ['.mp4']  # pre-processing outputs MP4s only


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        """
        Samples batches sequentially, always in the same order.

        :param data_source: data set to sample from
        :type data_source: Dataset
        :param batch_size: concurrent number of video streams
        :type batch_size: int
        """
        self.batch_size = batch_size
        self.samples_per_row = ceil(len(data_source) / batch_size)
        self.num_samples = self.samples_per_row * batch_size

    def __iter__(self):
        return (self.samples_per_row * i + j for j in range(self.samples_per_row) for i in range(self.batch_size))

    def __len__(self):
        return self.num_samples  # fake nb of samples, transparent wrapping around


class VideoCollate:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, batch: iter) -> torch.Tensor or list(torch.Tensor):
        """
        Puts each data field into a tensor with outer dimension batch size

        :param batch: samples from a Dataset object
        :type batch: list
        :return: temporal batch of frames of size (t, batch_size, *frame.size()), 0 <= t < T, most likely t = T - 1
        :rtype: tuple
        """
        if torch.is_tensor(batch[0]):
            return torch.cat(tuple(t.unsqueeze(0) for t in batch), 0).view(-1, self.batch_size, *batch[0].size())
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch).view(-1, self.batch_size)
        elif isinstance(batch[0], collections.Iterable):
            # if each batch element is not a tensor, then it should be a tuple
            # of tensors; in that case we collate each element in the tuple
            transposed = zip(*batch)
            return tuple(self.__call__(samples) for samples in transposed)

        raise TypeError(("batch must contain tensors, numbers, or lists; found {}"
                         .format(type(batch[0]))))


class VideoFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, video_index=False, shuffle=None):
        """
        Initialise a ``data.Dataset`` object for concurrent frame fetching from videos in a directory of folders of videos

        :param root: Data directory (train or validation folders path)
        :type root: str
        :param transform: image transform-ing object from ``torchvision.transforms``
        :type transform: object
        :param target_transform: label transformation / mapping
        :type target_transform: object
        :param video_index: if ``True``, the label will be the video index instead of target class
        :type video_index: bool
        :param shuffle: ``None``, ``'init'`` or ``True``
        :type shuffle: str
        """
        classes, class_to_idx = self._find_classes(root)
        video_paths = self._find_videos(root, classes)
        videos, frames, frames_per_video, frames_per_class = self._make_data_set(
            root, video_paths, class_to_idx, shuffle, video_index
        )

        self.root = root
        self.video_paths = video_paths
        self.videos = videos
        self.opened_videos = [[] for _ in videos]
        self.frames = frames
        self.frames_per_video = frames_per_video
        self.frames_per_class = frames_per_class
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.alternative_target = video_index
        self.shuffle = shuffle

    def __getitem__(self, frame_idx):
        if frame_idx == 0:
            self.free()
            if self.shuffle is True:
                self._shuffle()

        frame_idx %= self.frames  # wrap around indexing, if asking too much
        video_idx = bisect(self.videos, ((frame_idx,),))  # video to which frame_idx belongs
        (last, first), (path, target) = self.videos[video_idx]  # get video metadata
        frame = self._get_frame(frame_idx - first, video_idx, frame_idx == last)  # get frame from video
        if self.transform is not None:  # image processing
            frame = self.transform(frame)
        if self.target_transform is not None:  # target processing
            target = self.target_transform(target)

        if self.alternative_target: return frame, video_idx

        return frame, target

    def __len__(self):
        return self.frames

    def _get_frame(self, seek, video_idx, last):

        opened_video = None  # handle to opened target video
        if self.opened_videos[video_idx]:  # if handle(s) exists for target video
            current = self.opened_videos[video_idx]  # get handles list
            opened_video = next((ov for ov in current if ov[0] == seek), None)  # look for matching seek

        if opened_video is None:  # no (matching) handle found
            video_path = join(self.root, self.videos[video_idx][1][0])  # build video path
            video_file = FFmpegReader(video_path)  # get a video file pointer
            video_iter = video_file.nextFrame()  # get an iterator
            opened_video = [seek, islice(video_iter, seek, None), video_file]  # seek video and create o.v. item
            self.opened_videos[video_idx].append(opened_video)  # add opened video object to o.v. list

        opened_video[0] = seek + 1  # update seek pointer
        frame = next(opened_video[1])  # cache output frame
        if last:
            opened_video[2]._close()  # close video file (private method?!)
            self.opened_videos[video_idx].remove(opened_video)  # remove o.v. item

        return frame

    def free(self):
        """
        Frees all video files' pointers
        """
        for video in self.opened_videos:  # for every opened video
            for _ in range(len(video)):  # for as many times as pointers
                opened_video = video.pop()  # pop an item
                opened_video[2]._close()  # close the file

    def _shuffle(self):
        """
        Shuffles the video list
        by regenerating the sequence to sample sequentially
        """
        def _is_video_file(filename_):
            return any(filename_.endswith(extension) for extension in VIDEO_EXTENSIONS)

        root = self.root
        video_paths = self.video_paths
        class_to_idx = self.class_to_idx
        list_shuffle(video_paths)  # shuffle

        videos = list()
        frames_per_video = list()
        frames_counter = 0
        for filename in tqdm(video_paths, ncols=80):
            class_ = filename.split('/')[0]
            data_path = join(root, filename)
            if _is_video_file(data_path):
                video_meta = ffprobe(data_path)
                start_idx = frames_counter
                frames = int(video_meta['video'].get('@nb_frames'))
                frames_per_video.append(frames)
                frames_counter += frames
                item = ((frames_counter - 1, start_idx), (filename, class_to_idx[class_]))
                videos.append(item)

        sleep(0.5)  # allows for progress bar completion
        # update the attributes with the altered sequence
        self.video_paths = video_paths
        self.videos = videos
        self.frames = frames_counter
        self.frames_per_video = frames_per_video

    @staticmethod
    def _find_classes(data_path):
        classes = [d for d in listdir(data_path) if isdir(join(data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def _find_videos(root, classes):
        return [join(c, d) for c in classes for d in listdir(join(root, c))]

    @staticmethod
    def _make_data_set(root, video_paths, class_to_idx, init_shuffle, video_index):
        def _is_video_file(filename_):
            return any(filename_.endswith(extension) for extension in VIDEO_EXTENSIONS)

        if init_shuffle and not video_index:
            list_shuffle(video_paths)  # shuffle

        videos = list()
        frames_per_video = list()
        frames_per_class = [0] * len(class_to_idx)
        frames_counter = 0
        for filename in tqdm(video_paths, ncols=80):
            class_ = filename.split('/')[0]
            data_path = join(root, filename)
            if _is_video_file(data_path):
                video_meta = ffprobe(data_path)
                start_idx = frames_counter
                frames = int(video_meta['video'].get('@nb_frames'))
                frames_per_video.append(frames)
                frames_per_class[class_to_idx[class_]] += frames
                frames_counter += frames
                item = ((frames_counter - 1, start_idx), (filename, class_to_idx[class_]))
                videos.append(item)

        sleep(0.5)  # allows for progress bar completion
        return videos, frames_counter, frames_per_video, frames_per_class


def _test_video_folder():
    from textwrap import fill, indent

    batch_size = 5

    video_data_set = VideoFolder('small_data_set/')
    nb_of_classes = len(video_data_set.classes)
    print('There are', nb_of_classes, 'classes')
    print(indent(fill(' '.join(video_data_set.classes), 77), '   '))
    print('There are {} frames'.format(len(video_data_set)))
    print('Videos in the data set:', *video_data_set.videos, sep='\n')

    import inflect
    ordinal = inflect.engine().ordinal

    def print_list(my_list):
        for a, b in enumerate(my_list):
            print(a, ':', end=' [')
            print(*b, sep=',\n     ', end=']\n')

    # get first 3 batches
    n = ceil(len(video_data_set) / batch_size)
    print('Batch size:', batch_size)
    print('Frames per row:', n)
    for big_j in range(0, n, 90):
        batch = list()
        for j in range(big_j, big_j + 90):
            if j >= n: break  # there are no more frames
            batch.append(tuple(video_data_set[i * n + j][0] for i in range(batch_size)))
            batch[-1] = concatenate(batch[-1], 0)
        batch = concatenate(batch, 1)
        _show_numpy(batch, 1e-1)
        print(ordinal(big_j // 90 + 1), '90 batches of shape', batch.shape)
        print_list(video_data_set.opened_videos)

    print('Freeing resources')
    video_data_set.free()
    print_list(video_data_set.opened_videos)

    # get frames 50 -> 52
    batch = list()
    for i in range(50, 53):
        batch.append(video_data_set[i][0])
    _show_numpy(concatenate(batch, 1))
    print_list(video_data_set.opened_videos)


def _test_data_loader():
    big_t = 10
    batch_size = 5
    t = trn.Compose((trn.ToPILImage(), trn.ToTensor()))  # <-- add trn.CenterCrop(224) in between for training
    data_set = VideoFolder('small_data_set', t)
    my_loader = data.DataLoader(dataset=data_set, batch_size=batch_size * big_t, shuffle=False,
                                sampler=BatchSampler(data_set, batch_size), num_workers=0,
                                collate_fn=VideoCollate(batch_size))
    print('Is my_loader an iterator [has __next__()]:', isinstance(my_loader, collections.Iterator))
    print('Is my_loader an iterable [has __iter__()]:', isinstance(my_loader, collections.Iterable))
    my_iter = iter(my_loader)
    my_batch = next(my_iter)
    print('my_batch is a', type(my_batch), 'of length', len(my_batch))
    print('my_batch[0] is a', my_batch[0].type(), 'of size', tuple(my_batch[0].size()), '  # will 224, 224')
    _show_torch(_tile_up(my_batch), .2)
    for i in range(3): _show_torch(_tile_up(next(my_iter)), .2)


def _show_numpy(tensor: ndarray, zoom: float = 1.) -> None:
    """
    Display a ndarray image on screen

    :param tensor: image to visualise, of size (h, w, 1/3)
    :type tensor: ndarray
    :param zoom: zoom factor
    :type zoom: float
    """
    from PIL import Image
    shape = tuple(map(lambda s: round(s * zoom), tensor.shape))
    Image.fromarray(tensor).resize((shape[1], shape[0])).show()


def _show_torch(tensor: torch.FloatTensor, zoom: float = 1.) -> None:
    numpy_tensor = tensor.clone().mul(255).int().numpy().astype('u1').transpose(1, 2, 0)
    _show_numpy(numpy_tensor, zoom)


def _tile_up(temporal_batch):
    a = torch.cat(tuple(temporal_batch[0][:, i] for i in range(temporal_batch[0].size(1))), 2)
    a = torch.cat(tuple(a[j] for j in range(a.size(0))), 2)
    return a


if __name__ == '__main__':
    _test_video_folder()
    _test_data_loader()


__author__ = "Alfredo Canziani"
__credits__ = ["Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Production"  # "Prototype", "Development", or "Production"
__date__ = "Feb 17"
