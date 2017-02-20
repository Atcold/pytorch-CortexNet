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


def _is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def _find_classes(data_path):
    classes = [d for d in listdir(data_path) if isdir(join(data_path, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _make_data_set(data_path, classes, class_to_idx):
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


class VideoFolder(data.Dataset):
    def __init__(self, root, batch_size, transform=None, target_transform=None):
        # TODO: deal with batch_size and fake tot_nb_frames
        classes, class_to_idx = _find_classes(root)
        videos, frames = _make_data_set(root, classes, class_to_idx)

        self.root = root
        self.batch_size = batch_size
        self.videos = videos
        self.opened_videos = [[] for _ in videos]
        self.frames = frames
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def _get_frame(self, seek, video_idx):

        opened_video = None
        if self.opened_videos[video_idx]:
            # look for matching seek
            current = self.opened_videos[video_idx]
            opened_video = next((ov for ov in current if ov[0] + 1 == seek), None)  # unknown warning

        if opened_video is None:
            video_path = join(self.root, self.videos[video_idx][1][0])
            video_iter = vreader(video_path)
            opened_video = [seek, islice(video_iter, seek, None)]
            self.opened_videos[video_idx].append(opened_video)

        opened_video[0] = seek
        return next(opened_video[1])

    def __getitem__(self, frame_idx):
        video_idx = bisect(self.videos, ((frame_idx,),))
        (last, first), (path, target) = self.videos[video_idx]
        frame = self._get_frame(frame_idx - first, video_idx)
        if self.transform is not None:
            frame = self.transform(frame)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return frame, target

    def __len__(self):
        return self.frames


def test():
    from textwrap import fill, indent

    batch_size = 5

    video_data_set = VideoFolder('small_data_set/', batch_size=batch_size)
    nb_of_classes = len(video_data_set.classes)
    print('There are', nb_of_classes, 'classes')
    print(indent(fill(' '.join(video_data_set.classes), 77), '   '))
    print('There are {} frames'.format(len(video_data_set)))
    print('Videos in the data set:', *video_data_set.videos, sep='\n')

    from PIL import Image

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
    for j in range(3):
        batch = tuple(video_data_set[i * n + j][0] for i in range(5))
        batch = concatenate(batch, 0)
        Image.fromarray(batch).\
            resize((batch.shape[1]//2, batch.shape[0]//2)).\
            show(title='batch ' + str(j))
    print_list(video_data_set.opened_videos)

    pass

if __name__ == '__main__':
    test()
