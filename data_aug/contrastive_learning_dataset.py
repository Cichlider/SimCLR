from torchvision import transforms, datasets

from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, augmentation='baseline', s=1):
        """Build a SimCLR transform pipeline for a named augmentation setting."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        transform_ops = [transforms.RandomResizedCrop(size=size)]

        if augmentation != 'no_flip':
            transform_ops.append(transforms.RandomHorizontalFlip())

        if augmentation == 'light_color_jitter':
            light_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
            transform_ops.append(transforms.RandomApply([light_jitter], p=0.8))
        elif augmentation not in {'no_color_jitter', 'crop_only'}:
            transform_ops.append(transforms.RandomApply([color_jitter], p=0.8))

        if augmentation not in {'no_grayscale', 'crop_only'}:
            transform_ops.append(transforms.RandomGrayscale(p=0.2))

        if augmentation not in {'no_blur', 'crop_only'}:
            transform_ops.append(GaussianBlur(kernel_size=int(0.1 * size)))

        transform_ops.append(transforms.ToTensor())

        return transforms.Compose(transform_ops)

    def get_dataset(self, name, n_views, augmentation='baseline'):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32, augmentation),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96, augmentation),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
