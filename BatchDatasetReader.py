import numpy as np
import scipy.misc as misc
from PIL import Image, ImageOps, ImageEnhance
import random


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0  # don't use this time
    ratio = 0.25

    def __init__(self, records_list, image_options={}, augmentation=False):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file name records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self.prosess_image = True
        self.data_augmentation = augmentation
        self.__channels = True
        self._read_images()

    def _read_images(self):
        # self.__channels = True  # require gray immages
        # self.prosessimg = True
        # self.images = np.array([self._transform(filename['image']) for filename in self.files])
        # # self.images = np.expand_dims(self.images, 3)
        # self.__channels = True
        # self.prosessimg = False
        # self.annotations = np.array([self._transform(filename['annotation']) for filename in self.files])
        # self.annotations = np.expand_dims(self.annotations, 3)
        # print('self.images.shape:', self.images.shape)
        # print('self.annotations.shape:', self.annotations.shape)

        self.imageslist = [self._get_image(filename['image']) for filename in self.files]
        self.annotationslist = [self._get_image(filename['annotation']) for filename in self.files]
        print('----self.images.length:', len(self.imageslist))
        print('----self.annotations.length:', len(self.annotationslist))

    def _get_image(self, filename):
        image = Image.open(filename)

        return image


    def _transform(self, filename):
        image_object = Image.open(filename)
        crop_image = image_object.crop((270, 0, 3710, 2848))
        padding = (0, 296, 0, 296)
        pad_image = ImageOps.expand(crop_image, padding)  # width and height same size

        if self.__channels == False:
            image = np.array(pad_image.convert('L'))
        else:
            image = np.array(pad_image)

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size],
                                         interp='bicubic')  # bicubic interpolation resize image
        else:
            resize_image = image

        if self.prosess_image is True:
            resize_image = resize_image * (1.0 / 255)
            resize_image = per_image_standardization(resize_image)

        return np.array(resize_image)

    def _crop_resize_image(self, image, annotation):
        rate = [0.248, 0.25, 0.252]
        index = random.randint(0, 2)
        old_size = image.size
        new_size = tuple([int(x * rate[index]) for x in old_size])
        resize_image = image.resize(size=new_size, resample=3)
        resize_annotation = annotation.resize(size=new_size, resample=3)
        crop_image = resize_image.crop((66, 0, 930, 712))
        crop_annotation = resize_annotation.crop((66, 0, 930, 712))
        padding = (0, 76, 0, 76)
        image = ImageOps.expand(crop_image, padding)
        annotation = ImageOps.expand(crop_annotation, padding)
        if image.size != annotation.size:
            raise ValueError("Image and annotation size not equal !!!")

        # random crop image to size (640, 640)
        width, height = image.size
        resize = int(self.image_options["resize_size"])
        x = random.randint(0, width - resize - 1)
        y = random.randint(0, height - resize - 1)
        image = image.crop((x, y, x + resize, y + resize))
        annotation = annotation.crop((x, y, x + resize, y + resize))


        if self.data_augmentation is True:
            # light
            enh_bri = ImageEnhance.Brightness(image)
            brightness = round(random.uniform(0.8, 1.2), 2)
            image = enh_bri.enhance(brightness)

            # color
            enh_col = ImageEnhance.Color(image)
            color = round(random.uniform(0.8, 1.2), 2)
            image = enh_col.enhance(color)


            # contrast
            enh_con = ImageEnhance.Contrast(image)
            contrast = round(random.uniform(0.8, 1.2), 2)
            image = enh_con.enhance(contrast)
            #
            # enh_sha = ImageEnhance.Sharpness(image)
            # sharpness = round(random.uniform(0.8, 1.2), 2)
            # image = enh_sha.enhance(sharpness)

            method = random.randint(0, 7)
            # print(method)
            if method < 7:
                image = image.transpose(method)
                annotation = annotation.transpose(method)
            degree = random.randint(-5, 5)
            image = image.rotate(degree)
            annotation = annotation.rotate(degree)

        image_array = np.array(image)
        #standardization image
        if self.prosess_image is True:
            image_array = image_array * (1.0 / 255)
            # image_array = per_image_standardization(image_array)

        annotation_array = np.array(annotation)
        return np.array(image_array), annotation_array

    def get_records(self):
        return self.imageslist, self.annotationslist

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.imageslist):
            # Finished epoch
            # self.epochs_completed += 1
            # print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            # perm = np.arange(len(self.imageslist))
            # np.random.shuffle(perm)
            c = list(zip(self.imageslist, self.annotationslist))
            random.shuffle(c)
            self.imageslist, self.annotationslist = zip(*c)
            # self.imageslist = self.imageslist[perm]
            # self.annotationslist = self.annotationslist[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        image_batch = []
        annotation_batch = []
        for (image, annotation) in zip(self.imageslist[start:end], self.annotationslist[start:end]):
            img, annot = self._crop_resize_image(image, annotation)
            image_batch.append(img)
            annotation_batch.append(annot)
        return np.array(image_batch), np.expand_dims(np.array(annotation_batch), 3)

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.imageslist), size=[batch_size]).tolist()
        image = []
        annotation = []
        for index in indexes:
            img, annot = self._crop_resize_image(self.imageslist[index], self.annotationslist[index])
            image.append(img)
            annotation.append(annot)
        return np.array(image), np.expand_dims(np.array(annotation), 3)


def per_image_standardization(image):
    image = image.astype(np.float32, copy=False)
    mean = np.mean(image)
    stddev = np.std(image)
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.array(image.size, dtype=np.float32)))
    im = (image - mean) / adjusted_stddev
    return im