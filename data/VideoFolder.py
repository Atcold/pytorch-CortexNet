from math import ceil

import torch.utils.data as data
from os import listdir
from os.path import isdir, join

from itertools import islice

from numpy.core.multiarray import concatenate
from skvideo.io import vreader, ffprobe
from tqdm import tqdm
from time import sleep
from bisect import bisect

# Implement object from https://discuss.pytorch.org/t/loading-videos-from-folders-as-a-dataset-object/568

VIDEO_EXTENSIONS = ['.mp4']  # pre-processing outputs MP4s only








class VideoFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        """
        Initialise a data.Dataset object for concurrent frame fetching from videos in a directory of folders of videos

        :param root: Data directory (train or validation folders path)
        :type root: str
        :param transform: image transform-ing object from ``torchvision.transforms``
        :type transform: object
        :param target_transform: label transformation / mapping
        :type target_transform: object
        """
        classes, class_to_idx = self._find_classes(root)
        videos, frames = self._make_data_set(root, classes, class_to_idx)

        self.root = root
        self.videos = videos
        self.opened_videos = [[] for _ in videos]
        self.frames = frames
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, frame_idx):
        frame_idx %= self.frames  # wrap around indexing, if asking too much
        video_idx = bisect(self.videos, ((frame_idx,),))  # video to which frame_idx belongs
        (last, first), (path, target) = self.videos[video_idx]  # get video metadata
        frame = self._get_frame(frame_idx - first, video_idx, frame_idx == last)  # get frame from video
        if self.transform is not None:  # image processing
            frame = self.transform(frame)
        if self.target_transform is not None:  # target processing
            target = self.target_transform(target)

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
            video_iter = vreader(video_path)  # get an iterator
            opened_video = [seek, islice(video_iter, seek, None)]  # go to target seek and create o.v. object
            self.opened_videos[video_idx].append(opened_video)  # add opened video object to o.v. list

        opened_video[0] = seek + 1  # update seek pointer
        frame = next(opened_video[1])  # cache output frame
        if last: self.opened_videos[video_idx].remove(opened_video)  # remove exhausted iterator

        return frame

    def free(self):
        # TODO: close all video files and empty
        pass

    @staticmethod
    def _find_classes(data_path):
        classes = [d for d in listdir(data_path) if isdir(join(data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def _make_data_set(data_path, classes, class_to_idx):
        def _is_video_file(filename_):
            return any(filename_.endswith(extension) for extension in VIDEO_EXTENSIONS)
        videos = list()
        frames = 0
        for class_ in tqdm(classes, ncols=80):
            class_path = join(data_path, class_)
            for filename in listdir(class_path):
                if _is_video_file(filename):
                    video_path = join(class_path, filename)
                    video_meta = ffprobe(video_path)
                    start_idx = frames
                    frames += int(video_meta['video'].get('@nb_frames'))
                    item = ((frames - 1, start_idx), (join(class_, filename), class_to_idx[class_]))
                    videos.append(item)

        sleep(0.1)  # allows for progress bar completion
        return videos, frames


def test():
    from textwrap import fill, indent

    batch_size = 5

    video_data_set = VideoFolder('small_data_set/')
    nb_of_classes = len(video_data_set.classes)
    print('There are', nb_of_classes, 'classes')
    print(indent(fill(' '.join(video_data_set.classes), 77), '   '))
    print('There are {} frames'.format(len(video_data_set)))
    print('Videos in the data set:', *video_data_set.videos, sep='\n')

    from PIL import Image
    import inflect
    ordinal = inflect.engine().ordinal

    # get frames 50 -> 52
    # for i in range(50, 53):
    #     Image.fromarray(video_data_set[i][0]).show()
    # print(video_data_set.opened_videos)

    def print_list(my_list):
        for a, b in enumerate(my_list):
            print(a, ':', b)

    # Image.fromarray(video_data_set[0][0]).show()
    # print_list(video_data_set.opened_videos)
    #
    # Image.fromarray(video_data_set[252][0]).show()
    # print_list(video_data_set.opened_videos)

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
        print(batch.shape, 50*'-')
        Image.fromarray(batch).\
            resize((batch.shape[1]//10, batch.shape[0]//10))
        print(ordinal(big_j // 90 + 1), '90 batches of shape', batch.shape)
        print_list(video_data_set.opened_videos)

    pass

if __name__ == '__main__':
    test()
