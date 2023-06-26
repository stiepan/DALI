from custom_rand_augment import rand_augment
from mixup_cutmix import mixup_cutmix, mixup_cutmix_one_hot_encoding
from random_erasure import random_erasure

from nvidia.dali import fn, pipeline_def

img_shape = (224, 224)
random_area = (0.08, 1.0)
random_aspect_ratio = (0.75, 1.3333)
data_path = "/home/ktokarski/DALI_extra/db/single/jpeg"
num_classes = 20


def create_pipeline(img_shape=img_shape, random_area=random_area,
                    random_aspect_ratio=random_aspect_ratio, num_classes=num_classes,
                    data_path=data_path, batch_size=16, mixup_alpha=0.8, cutmix_alpha=1.,
                    mixup_prob=1., mixup_switch_prob=0.5, mixup_cutmix_mode="elem",
                    label_smoothing=0.1, re_count=1, re_prob=0.25, color_range=255,
                    re_mode="random", ra_n=2, ra_m=9, ra_num_bins=11, ra_prob=0.5, ra_std=0.5,
                    ra_max_translation_offset=100):

    if not mixup_cutmix_mode in ("batch", "elem"):
        raise Exception(f"Unsupported `mixup_mode`: `{mixup_cutmix_mode}`.")
    uniform_mix_cut = mixup_cutmix_mode == "batch"

    use_rand_augment = ra_n > 0
    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    use_random_erasure = re_prob > 0

    @pipeline_def(enable_conditionals=True, batch_size=batch_size, device_id=0, num_threads=4)
    def pipeline():
        image, label = fn.readers.file(file_root=data_path, random_shuffle=True)
        image = fn.decoders.image(image, device="mixed")
        image = fn.random_resized_crop(image, size=img_shape, random_area=random_area,
                                       random_aspect_ratio=random_aspect_ratio)
        image = fn.flip(image, horizontal=fn.random.coin_flip())
        if use_rand_augment:
            image = rand_augment(image, n=ra_n, m=ra_m, num_magnitude_bins=ra_num_bins,
                                 apply_prob=ra_prob, std_dev=ra_std,
                                 max_translate_abs=ra_max_translation_offset)
        if use_mixup_cutmix:
            label = mixup_cutmix_one_hot_encoding(label_smoothing, num_classes, label)
            image, label = mixup_cutmix(mixup_prob, mixup_switch_prob, mixup_alpha, cutmix_alpha,
                                        uniform_mix_cut, batch_size, img_shape, image, label)
        if use_random_erasure:
            image = random_erasure(image, prob=re_prob, mode=re_mode, count=re_count,
                                   color_range=color_range)
        return image, label

    return pipeline()


p = create_pipeline()
p.build()
images, labels = p.run()