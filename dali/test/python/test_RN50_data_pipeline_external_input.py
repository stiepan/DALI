# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.external_source as ext_source
import os
import glob
import argparse
import time
from test_utils import get_dali_extra_path

from os import listdir, path
import numpy as np
from PIL import Image
import cv2


import concurrent.futures
import urllib.request


class DataSet(object):

    def __init__(self, data_path):
        self.data_path = data_path
        files = self.get_files_list(self.data_path)
        self.classes_no = len(files)
        counter = 0
        self.files_map = {}
        for class_no, (class_name, samples) in enumerate(files):
            for sample in samples:
                self.files_map[counter] = (sample, class_no)
                counter += 1
        self.size = counter

    @classmethod
    def get_files_list(cls, data_path):
        dirs = [
            (_dir, [
                file_path
                for file_path
                in map(lambda name: path.join(_dir_path, name), listdir(_dir_path))
                if path.isfile(file_path)
            ])
            for (_dir, _dir_path)
            in map(lambda name: (name, path.join(data_path, name)), listdir(data_path))
            if path.isdir(_dir_path)
        ]
        dirs.sort(key=lambda dir_files: dir_files[0])
        for _, files in dirs:
            files.sort()
        return dirs

    def __len__(self):
        return self.size

    def __getitem__(self, indx):
        assert(type(indx) == int and 0 <= indx < self.size)
        return self.files_map[indx]


class BaseDataLoader(object):

    DATASET_CLASS = DataSet

    def __init__(self, dataset_dir, batch_size):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.ds = self.get_dataset()

    def get_dataset(self):
        return self.DATASET_CLASS(self.dataset_dir)


class DataLoader(BaseDataLoader):

    def read_img(self, file_name):
        with open(file_name, 'rb') as f:
            return np.fromstring(f.read(), dtype=np.uint8)

    def get_iter(self):
        sampled = 0
        ds_len = len(self.ds)
        while True:
            files = [self.ds[i % ds_len] for i in range(sampled, sampled + self.batch_size)]
            sampled += self.batch_size
            imgs = [self.read_img(file_path) for file_path, label in files]
            labels = [np.array([label]) for file_path, label in files]
            yield imgs, labels


class PILImageLoader(DataLoader):

    def read_img(self, file_name):
        img = Image.open(file_name)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)


# class DataLoaderParallel(BaseDataLoader):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

#     def read_img(self, p):
#         file_name, label = p
#         img = Image.open(file_name)
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#         return np.array(img), np.array([label])

#     def get_sample(self, iter_no, batch_offset):
#         ds_len = len(self.ds)
#         i = iter_no * self.batch_size + 2 * batch_offset
#         return list(self.executor.map(self.read_img, [self.ds[(i + j) % ds_len] for j in range(2)]))


class DataLoaderParallelOpenCv(BaseDataLoader):

    # MINI_BATCH = 1

    def __init__(self, *args, **kwargs):
        print("dddddddd")
        super().__init__(*args, **kwargs)

    def read_img(self, p):
        file_name, label = p
        img = cv2.imread(file_name)
        return img, np.array([label])

    def get_sample(self, i):
        ds_len = len(self.ds)
        return self.read_img(self.ds[i % ds_len])


dlp = None
# def get_sample(iter_no, batch_offset):
#     global dlp
#     if dlp is None:
#         dlp = DataLoaderParallel("/home/ktokarski/imagenet_k/subset_train/", BATCH_SIZE)
#     return dlp.get_sample(iter_no, batch_offset)


def xdd(inh=ext_source.DamtaSemmmt):
    class DamtaSemt(inh):

        def __init__(self):
            super().__init__()
            self.dlp = DataLoaderParallelOpenCv("/home/ktokarski/imagenet_k/subset_train/", BATCH_SIZE)

        def __call__(self, idx):
            return self.dlp.get_sample(idx)

    return DamtaSemt()


class DamtaSemt(ext_source.DamtaSemmmt):

        def __init__(self):
            super().__init__()
            self.dlp = DataLoaderParallelOpenCv("/home/ktokarski/imagenet_k/subset_train/", BATCH_SIZE)

        def __call__(self, idx):
            return self.dlp.get_sample(idx)


# class ParallelDataLoader(PILImageLoader):

#     DATASET_CLASS = DataSet

#     def __init__(self, dataset_dir, batch_size, workers_no):
#         super().__init__(dataset_dir, batch_size)
#         self.workers_no = workers_no
#         self.pool = Pool(workers_no) if __name__ == '__main__' else None

#     def get_iter(self):
#         sampled = 0
#         ds_len = len(self.ds)
#         while sampled < ds_len:
#             files = [self.ds[i % ds_len] for i in range(sampled, sampled + self.batch_size)]
#             imgs = self.pool.map(self.read_img,  [file_path for file_path, label in files])
#             labels = [np.array([label]) for file_path, label in files]
#             yield imgs, labels


class BaseCommonPipeline(Pipeline):
    def __init__(self, data_paths, num_shards, batch_size, num_threads, device_id, prefetch, fp16, random_shuffle, nhwc,
                 dont_use_mmap, decoder_type, decoder_cache_params, reader_queue_depth, shard_id):
        super().__init__(batch_size, num_threads, device_id, random_shuffle, prefetch_queue_depth=prefetch)
        self.res = ops.RandomResizedCrop(device="gpu", size =(224,224))
        layout = types.NHWC
        out_type = types.FLOAT
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=out_type,
                                            output_layout=layout,
                                            crop=(224, 224),
                                            mean=[125, 125, 125],
                                            std=[255, 255, 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def base_define_graph(self, images, labels):
        rng = self.coin()
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return output, labels


class DALIDecodePipeline(BaseCommonPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decode_gpu = ops.ImageDecoder(device="cpu", output_type=types.RGB)

    def base_define_graph(self, inputs, labels):
        images = self.decode_gpu(inputs)
        return super().base_define_graph(images.gpu(), labels)


class FileReadPipeline(DALIDecodePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cache_enabled = kwargs['decoder_cache_params']['cache_enabled']
        self.input = ops.FileReader(file_root = kwargs['data_paths'][0],
                                    shard_id = kwargs['shard_id'],
                                    num_shards = kwargs['num_shards'],
                                    random_shuffle = kwargs['random_shuffle'],
                                    dont_use_mmap = kwargs['dont_use_mmap'],
                                    stick_to_shard = cache_enabled,
                                    #skip_cached_images = cache_enabled,
                                    prefetch_queue_depth = kwargs['reader_queue_depth'])

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


class DataLoaderMixIn(object):

    DATALOADER_CLASS = DataLoader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loader = self.get_data_loader(**kwargs)
        self.input = ops.ExternalSource(self.loader.get_iter, num_outputs=2)

    def get_data_loader(self, **kwargs):
        return self.DATALOADER_CLASS(kwargs['data_paths'][0], kwargs['batch_size'])

    def define_graph(self):
        images, labels = self.input()
        return self.base_define_graph(images, labels)

    def epoch_size(self, *args, **kwargs):
        return len(self.loader.ds)


class ExternalInputPipeline(DataLoaderMixIn, DALIDecodePipeline):
    pass


class ExternalPythonDecodePipeline(DataLoaderMixIn, BaseCommonPipeline):
    DATALOADER_CLASS = PILImageLoader

    def define_graph(self):
        images, labels = self.input()
        return self.base_define_graph(images.gpu(), labels)


# class ExternalPythonDecodePipelineParallel(BaseCommonPipeline):
#     DATALOADER_CLASS = DataLoader

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.loader = self.get_data_loader(**kwargs)
#         self.input = ops.ExternalSource(get_sample, num_outputs=2, batch_size=kwargs['batch_size'])

#     def get_data_loader(self, **kwargs):
#         return self.DATALOADER_CLASS(kwargs['data_paths'][0], kwargs['batch_size'])

#     def define_graph(self):
#         images, labels = self.input()
#         return self.base_define_graph(images.gpu(), labels)

#     def epoch_size(self, *args, **kwargs):
#         return len(self.loader.ds)


class ExternalOpenCVDecodePipelineParallel(BaseCommonPipeline):
    DATALOADER_CLASS = DataLoader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loader = self.get_data_loader(**kwargs)
        damta_set = DamtaSemt()
        self.input = ops.ExternalSource(damta_set, num_outputs=2, batch_size=kwargs['batch_size'], no_copy=True)

    def get_data_loader(self, **kwargs):
        return self.DATALOADER_CLASS(kwargs['data_paths'][0], kwargs['batch_size'])

    def define_graph(self):
        images, labels = self.input()
        return self.base_define_graph(images.gpu(), labels)

    def epoch_size(self, *args, **kwargs):
        return len(self.loader.ds)

if __name__ == "__main__":
    test_data = {
        # ExternalPythonDecodePipeline: [["/home/ktokarski/imagenet_k/subset_train/"]],
        ExternalOpenCVDecodePipelineParallel: [["/home/ktokarski/imagenet_k/subset_train/"]],
        # ExternalPythonDecodePipelineParallel: [["/home/ktokarski/imagenet_k/subset_train/"]],
        # ExternalInputPipeline: [["/home/ktokarski/imagenet_k/subset_train/"]],
        FileReadPipeline: [["/home/ktokarski/imagenet_k/subset_train/"]],
    }

    data_root = get_dali_extra_path()

    small_test_data = {
        ExternalInputPipeline: [[os.path.join(data_root, "db/single/jpeg/")]],
        FileReadPipeline: [[os.path.join(data_root, "db/single/jpeg/")]],
    }

    parser = argparse.ArgumentParser(description='Test nvJPEG based RN50 augmentation pipeline with different datasets')
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs run in parallel by this test (default: 1)')
    parser.add_argument('-b', '--batch', default=1024, type=int, metavar='N',
                        help='batch size (default: 1024)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 3)')
    parser.add_argument('--prefetch', default=2, type=int, metavar='N',
                        help='prefetch queue depth (default: 2)')
    parser.add_argument('--separate_queue', action='store_true',
                        help='Use separate queues executor')
    parser.add_argument('--cpu_size', default=2, type=int, metavar='N',
                        help='cpu prefetch queue depth (default: 2)')
    parser.add_argument('--gpu_size', default=2, type=int, metavar='N',
                        help='gpu prefetch queue depth (default: 2)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run fp16 pipeline')
    parser.add_argument('--nhwc', action='store_true',
                        help='Use NHWC data instead of default NCHW')
    parser.add_argument('-i', '--iters', default=-1, type=int, metavar='N',
                        help='Number of iterations to run (default: -1 - whole data set)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='Number of epochs to run')
    parser.add_argument('--decoder_type', default='', type=str, metavar='N',
                        help='split, roi, roi_split, cached (default: regular nvjpeg)')
    parser.add_argument('--cache_size', default=0, type=int, metavar='N',
                        help='Cache size (in MB)')
    parser.add_argument('--cache_threshold', default=0, type=int, metavar='N',
                        help='Cache threshold')
    parser.add_argument('--cache_type', default='none', type=str, metavar='N',
                        help='Cache type')
    parser.add_argument('--reader_queue_depth', default=1, type=int, metavar='N',
                        help='prefetch queue depth (default: 1)')
    parser.add_argument('--read_shuffle', action='store_true',
                        help='Shuffle data when reading')
    parser.add_argument('--disable_mmap', action='store_true',
                        help='Disable mmap for DALI readers. Used for network filesystem tests.')
    parser.add_argument('-s', '--small', action='store_true',
                        help='use small dataset, DALI_EXTRA_PATH needs to be set')
    parser.add_argument('--number_of_shards', default=None, type=int, metavar='N',
                        help='Number of shards in the dataset')
    parser.add_argument('--assign_gpu', default=None, type=int, metavar='N',
                        help='Assign a given GPU. Cannot be used with --gpus')
    parser.add_argument('--assign_shard', default=0, type=int, metavar='N',
                        help='Assign a given shard id. If used with --gpus, it assigns the first GPU to this id and next GPUs get consecutive ids')
    parser.add_argument('--simulate_N_gpus', default=None, type=int, metavar='N',
                        help='Used to simulate small shard as it would be in a multi gpu setup with this number of gpus. If provided, each gpu will see a shard size as if we were in a multi gpu setup with this number of gpus',
                        dest='number_of_shards')
    parser.add_argument('--remove_default_pipeline_paths', action='store_true',
                        help="For all data pipeline types, remove the default values")
    parser.add_argument('--file_read_pipeline_paths', default=None, type=str, metavar='N',
                        help='Add custom FileReadPipeline paths. Separate multiple paths by commas')
    parser.add_argument('--mxnet_reader_pipeline_paths', default=None, type=str, metavar='N',
                        help='Add custom MXNetReaderPipeline paths. For a given path, a .rec and .idx extension will be appended. Separate multiple paths by commas')
    parser.add_argument('--caffe_read_pipeline_paths', default=None, type=str, metavar='N',
                        help='Add custom CaffeReadPipeline paths. Separate multiple paths by commas')
    parser.add_argument('--caffe2_read_pipeline_paths', default=None, type=str, metavar='N',
                        help='Add custom Caffe2ReadPipeline paths. Separate multiple paths by commas')
    parser.add_argument('--tfrecord_pipeline_paths', default=None, type=str, metavar='N',
                        help='Add custom TFRecordPipeline paths. For a given path, a second path with an .idx extension will be added for the required idx file(s). Separate multiple paths by commas')
    args = parser.parse_args()

    N = args.gpus             # number of GPUs
    GPU_ID = args.assign_gpu
    DALI_SHARD = args.assign_shard
    BATCH_SIZE = args.batch   # batch size
    LOG_INTERVAL = args.print_freq
    WORKERS = args.workers
    PREFETCH = args.prefetch
    if args.separate_queue:
        PREFETCH = {'cpu_size': args.cpu_size , 'gpu_size': args.gpu_size}
    FP16 = args.fp16
    NHWC = args.nhwc

    if args.remove_default_pipeline_paths:
        for pipe_name in test_data.keys():
            test_data[pipe_name] = []

    if args.file_read_pipeline_paths:
        paths = args.file_read_pipeline_paths.split(',')
        for path in paths:
            test_data[FileReadPipeline].append([path])

    DECODER_TYPE = args.decoder_type
    CACHED_DECODING = DECODER_TYPE == 'cached'
    DECODER_CACHE_PARAMS = {}
    DECODER_CACHE_PARAMS['cache_enabled'] = CACHED_DECODING
    if CACHED_DECODING:
        DECODER_CACHE_PARAMS['cache_type'] = args.cache_type
        DECODER_CACHE_PARAMS['cache_size'] = args.cache_size
        DECODER_CACHE_PARAMS['cache_threshold'] = args.cache_threshold
    READER_QUEUE_DEPTH = args.reader_queue_depth
    NUMBER_OF_SHARDS = N if args.number_of_shards == None else args.number_of_shards
    STICK_TO_SHARD = True if CACHED_DECODING else False
    SKIP_CACHED_IMAGES = True if CACHED_DECODING else False

    READ_SHUFFLE=args.read_shuffle

    DISABLE_MMAP=args.disable_mmap

    SMALL_DATA_SET = args.small
    if SMALL_DATA_SET:
        test_data = small_test_data

    print("GPUs: {}, batch: {}, workers: {}, prefetch depth: {}, loging interval: {}, fp16: {}, NHWC: {}, READ_SHUFFLE: {}, DISABLE_MMAP: {}, small dataset: {}, GPU ID: {}, shard number: {}, number of shards {}"
        .format(N, BATCH_SIZE, WORKERS, PREFETCH, LOG_INTERVAL, FP16, NHWC, READ_SHUFFLE, DISABLE_MMAP, SMALL_DATA_SET, GPU_ID, DALI_SHARD, NUMBER_OF_SHARDS))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_last_n = 0
        self.max_val = 0

    def update(self, val, n=1):
        self.val = val
        self.max_val = max(self.max_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    for pipe_name in test_data.keys():
        data_set_len = len(test_data[pipe_name])
        for i, data_set in enumerate(test_data[pipe_name]):
            if GPU_ID is None:
                pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=WORKERS, device_id=n,
                                num_shards=NUMBER_OF_SHARDS, data_paths=data_set, prefetch=PREFETCH, fp16=FP16, random_shuffle=READ_SHUFFLE,
                                dont_use_mmap=DISABLE_MMAP, nhwc=NHWC, decoder_type=DECODER_TYPE, decoder_cache_params=DECODER_CACHE_PARAMS,
                                reader_queue_depth=READER_QUEUE_DEPTH, shard_id=DALI_SHARD+n) for n in range(N)]
            else:
                pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=WORKERS, device_id=GPU_ID,
                                num_shards=NUMBER_OF_SHARDS, data_paths=data_set, prefetch=PREFETCH, fp16=FP16, random_shuffle=READ_SHUFFLE,
                                dont_use_mmap=DISABLE_MMAP, nhwc=NHWC, decoder_type=DECODER_TYPE, decoder_cache_params=DECODER_CACHE_PARAMS,
                                reader_queue_depth=READER_QUEUE_DEPTH, shard_id=DALI_SHARD)]
            [pipe.build() for pipe in pipes]

            if args.iters < 0:
                iters = pipes[0].epoch_size("Reader")
                assert(all(pipe.epoch_size("Reader") == iters for pipe in pipes))
                iters_tmp = iters
                iters = iters // BATCH_SIZE
                if iters_tmp != iters * BATCH_SIZE:
                    iters += 1
                iters_tmp = iters

                iters = iters // NUMBER_OF_SHARDS
                if iters_tmp != iters * NUMBER_OF_SHARDS:
                    iters += 1
            else:
                iters = args.iters

            print("RUN {0}/{1}: {2}".format(i + 1,
                                            data_set_len, pipe_name.__name__))
            print(data_set)
            end = time.time()
            for i in range(args.epochs):
                if i == 0:
                    print("Warm up")
                else:
                    print("Test run " + str(i))
                data_time = AverageMeter()
                for j in range(iters):
                    for pipe in pipes:
                        pipe.run()
                    data_time.update(time.time() - end)
                    if j % LOG_INTERVAL == 0:
                        print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s]"
                                .format(pipe_name.__name__, j + 1, iters, data_time.avg, data_time.max_val, N * BATCH_SIZE / data_time.avg))
                    end = time.time()

            print("OK {0}/{1}: {2}".format(i + 1,
                                        data_set_len, pipe_name.__name__))
