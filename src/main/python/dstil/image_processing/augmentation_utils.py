from enum import Enum
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

class AugmentedImage(object):
    def __init__(self, image, title, image_height, image_width):

        # TODO: Add parameters to set the size of the output images, set to the same size
        # as the original image for now.
        self.image = image #ia.imresize_single_image(image, (image_height, image_width))
        self.title = title
        self.cell_height = image.shape[0]
        self.cell_width = image.shape[1]

    def draw(self):
        image_cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8) + 255
        image_cell[0:self.image.shape[0], 0:self.image.shape[1], :] = self.image
        return image_cell

class Detection(Enum):
    FACE = 1
    NO_FACE = 2

class AugmentationParams:
    def __init__(self, augmentation_value, detection_tag):
        self.augmentation_value = augmentation_value
        self.detection_tag = detection_tag

class Augmentation(object):
    def __init__(self, seed, name, augmentation_params):
        self.seed = seed
        self.name = name
        self.augmentation_params = augmentation_params
        self.augmented_images = []
        self.augmentation_data = []

    def logic(self):
        raise NotImplementedError("Please Implement this method")

    def augment(self, image, augmentation_logic):
        ia.seed(self.seed)
        augmentation_logic
        self.generate_export_data()

    def generate_export_data(self):
        for (title, augmentation, detection_tag) in self.augmentation_data:
            output_title = "%s_%s_%s" % (self.name, title.title().replace(" ", ""), detection_tag.name.lower().title().replace("_", ""))
            self.augmented_images. append(AugmentedImage(augmentation, output_title, 256, 256))

class Crop(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Crop", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Crop(px = param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Crop, self).augment(image, self.logic(image))

class FlipLr(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "FlipLr", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Fliplr(param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(FlipLr, self).augment(image, self.logic(image))

class FlipUd(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "FlipUd", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Flipud(param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(FlipUd, self).augment(image, self.logic(image))

class Pad(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Pad", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Pad(px=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Pad, self).augment(image, self.logic(image))

class SuperPixels(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "SuperPixels", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Superpixels(p_replace=param.augmentation_value, n_segments=100).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(SuperPixels, self).augment(image, self.logic(image))

class Invert(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Invert", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Invert(p=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Invert, self).augment(image, self.logic(image))

class InvertPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "InvertPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Invert(p=param.augmentation_value, per_channel=True).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(InvertPerChannel, self).augment(image, self.logic(image))

class Add(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Add", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Add(param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Add, self).augment(image, self.logic(image))

class AddPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AddPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append(["%d-%d" % (param.augmentation_value[0], param.augmentation_value[1]), iaa.Add(param.augmentation_value, per_channel=True).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AddPerChannel, self).augment(image, self.logic(image))

class AddToHueAndSaturation(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AddToHueAndSaturation", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.AddToHueAndSaturation(param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AddToHueAndSaturation, self).augment(image, self.logic(image))

class Multiply(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Multiply", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Multiply(param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Multiply, self).augment(image, self.logic(image))

class MultiplyPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 1, "MultiplyPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append(["%d-%d" % (param.augmentation_value[0], param.augmentation_value[1]), iaa.Multiply(param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(MultiplyPerChannel, self).augment(image, self.logic(image))

class GaussianBlur(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "GaussianBlur", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.GaussianBlur(sigma=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(GaussianBlur, self).augment(image, self.logic(image))

class AverageBlur(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AverageBlur", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.AverageBlur(k=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AverageBlur, self).augment(image, self.logic(image))

class MedianBlur(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "MedianBlur", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.MedianBlur(k=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(MedianBlur, self).augment(image, self.logic(image))

class BilateralBlur(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "BilateralBlur", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.BilateralBlur(d=param.augmentation_value, sigma_color=250, sigma_space=250).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(BilateralBlur, self).augment(image, self.logic(image))

class Sharpen(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Sharpen", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Sharpen(alpha=1, lightness=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Sharpen, self).augment(image, self.logic(image))

class Emboss(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Emboss", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Emboss(alpha=1, strength=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Emboss, self).augment(image, self.logic(image))

class AdditiveGaussianNoise(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AdditiveGaussianNoise", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.AdditiveGaussianNoise(scale=param.augmentation_value * 255).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AdditiveGaussianNoise, self).augment(image, self.logic(image))

class AdditiveGaussianNoisePerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AdditiveGaussianNoisePerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.AdditiveGaussianNoise(scale=param.augmentation_value * 255, per_channel=True).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AdditiveGaussianNoisePerChannel, self).augment(image, self.logic(image))

class Dropout(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Dropout", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Dropout(p=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Dropout, self).augment(image, self.logic(image))

class DropoutPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "DropoutPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Dropout(p=param.augmentation_value, per_channel=True).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(DropoutPerChannel, self).augment(image, self.logic(image))

class CoarseDropout(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 3, "CoarseDropout", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.CoarseDropout(p=0.2, size_percent=param.augmentation_value, min_size=2).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(CoarseDropout, self).augment(image, self.logic(image))

class CoarseDropoutPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "CoarseDropoutPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.CoarseDropout(p=0.2, size_percent=param.augmentation_value, min_size=2, per_channel=True).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(CoarseDropoutPerChannel, self).augment(image, self.logic(image))

class SaltAndPepper(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "SaltAndPepper", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.SaltAndPepper(p=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(SaltAndPepper, self).augment(image, self.logic(image))

class Salt(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Salt", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Salt(p=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Salt, self).augment(image, self.logic(image))

class Pepper(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Pepper", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Pepper(p=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Pepper, self).augment(image, self.logic(image))

class CoarseSaltAndPepper(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "CoarseSaltAndPepper", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.CoarseSaltAndPepper(p=0.2, size_percent=param.augmentation_value, min_size=2).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(CoarseSaltAndPepper, self).augment(image, self.logic(image))

class CoarseSalt(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "CoarseSalt", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.CoarseSalt(p=0.2, size_percent=param.augmentation_value, min_size=2).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(CoarseSalt, self).augment(image, self.logic(image))

class CoarsePepper(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "CoarsePepper", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.CoarsePepper(p=0.2, size_percent=param.augmentation_value, min_size=2).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(CoarsePepper, self).augment(image, self.logic(image))

class ContrastNormalization(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "ContrastNormalization", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.ContrastNormalization(alpha=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(ContrastNormalization, self).augment(image, self.logic(image))

class ContrastNormalizationPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "ContrastNormalizationPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append(["%d-%d" % (param.augmentation_value[0], param.augmentation_value[1]), iaa.ContrastNormalization(alpha=param.augmentation_value, per_channel=True).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(ContrastNormalizationPerChannel, self).augment(image, self.logic(image))

class Grayscale(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "Grayscale", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Grayscale(alpha=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(Grayscale, self).augment(image, self.logic(image))

class PerspectiveTransform(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 6, "PerspectiveTransform", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.PerspectiveTransform(scale=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(PerspectiveTransform, self).augment(image, self.logic(image))

class PiecewiseAffine(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "PiecewiseAffine", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.PiecewiseAffine(scale=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(PiecewiseAffine, self).augment(image, self.logic(image))

class AffineScale(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AffineScale", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Affine(scale=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AffineScale, self).augment(image, self.logic(image))

class AffineTranslate(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AffineTranslate", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append(["%d-%d" % (param.augmentation_value[0], param.augmentation_value[1]), iaa.Affine(translate_px={"x": param.augmentation_value[0], "y": param.augmentation_value[1]}).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AffineTranslate, self).augment(image, self.logic(image))

class AffineRotate(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AffineRotate", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Affine(rotate=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AffineRotate, self).augment(image, self.logic(image))

class AffineShear(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AffineShear", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Affine(shear=param.augmentation_value).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AffineShear, self).augment(image, self.logic(image))

class AffineCval(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AffineCval", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value * 255), iaa.Affine(translate_px=-32, cval=int(param.augmentation_value * 255), mode="constant").to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AffineCval, self).augment(image, self.logic(image))

class ElasticTransformation(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 1, "ElasticTransformation", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.ElasticTransformation(alpha=param.augmentation_value, sigma=0.2).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(ElasticTransformation, self).augment(image, self.logic(image))

class AlphaWithEdgeDetect(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 0, "AlphaWithEdgeDetect", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.Alpha(factor=param.augmentation_value, first=iaa.EdgeDetect(1.0)).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AlphaWithEdgeDetect, self).augment(image, self.logic(image))

class AlphaWithEdgeDetectPerChannel(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 4, "AlphaWithEdgeDetectPerChannel", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append(["%d-%d" % (param.augmentation_value[0], param.augmentation_value[1]), iaa.Alpha(factor=param.augmentation_value, first=iaa.EdgeDetect(1.0), per_channel = 0.5).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(AlphaWithEdgeDetectPerChannel, self).augment(image, self.logic(image))

class SimplexNoiseAlphaWithEdgeDetect(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 15, "SimplexNoiseAlphaWithEdgeDetect", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.SimplexNoiseAlpha(first=iaa.EdgeDetect(1.0)).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(SimplexNoiseAlphaWithEdgeDetect, self).augment(image, self.logic(image))

class FrequencyNoiseAlphaWithEdgeDetect(Augmentation):
    def __init__(self, augmentation_params):
        Augmentation.__init__(self, 9, "FrequencyNoiseAlphaWithEdgeDetect", augmentation_params)

    def logic(self, image):
        for param in self.augmentation_params:
            self.augmentation_data.append([str(param.augmentation_value), iaa.FrequencyNoiseAlpha(exponent=param.augmentation_value, first=iaa.EdgeDetect(1.0), size_px_max=16, upscale_method="linear", sigmoid=False).to_deterministic().augment_image(image), param.detection_tag])

    def augment(self, image):
        super(FrequencyNoiseAlphaWithEdgeDetect, self).augment(image, self.logic(image))

augmentation_objects = [
        Crop([
            AugmentationParams((2, 0, 0, 0), Detection.FACE),
            AugmentationParams((0, 8, 8, 0), Detection.FACE),
            AugmentationParams((4, 0, 16, 4), Detection.FACE),
            AugmentationParams((8, 0, 0, 32), Detection.FACE),
            AugmentationParams((32, 64, 0, 0), Detection.FACE)
        ]),
        FlipLr([
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1), Detection.FACE)
        ]),
        FlipUd([
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1), Detection.FACE)
        ]),
        Pad([
            AugmentationParams((2, 0, 0, 0), Detection.FACE),
            AugmentationParams((0, 8, 8, 0), Detection.FACE),
            AugmentationParams((4, 0, 16, 4), Detection.FACE),
            AugmentationParams((8, 0, 0, 32), Detection.FACE),
            AugmentationParams((32, 64, 0, 0), Detection.FACE)
        ]),
        SuperPixels([
            AugmentationParams((0.0), Detection.FACE),
            AugmentationParams((0.25), Detection.FACE),
            AugmentationParams((0.5), Detection.NO_FACE),
            AugmentationParams((0.75), Detection.NO_FACE),
            AugmentationParams((1.0), Detection.NO_FACE)
        ]),
        Invert([
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((1), Detection.NO_FACE),
            AugmentationParams((1), Detection.NO_FACE),
            AugmentationParams((1), Detection.NO_FACE)
        ]),
        InvertPerChannel([
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((0.5), Detection.NO_FACE),
            AugmentationParams((0.5), Detection.NO_FACE),
            AugmentationParams((0.5), Detection.NO_FACE)
        ]),
        Add([
            AugmentationParams((-45), Detection.FACE),
            AugmentationParams((-25), Detection.FACE),
            AugmentationParams((0), Detection.NO_FACE),
            AugmentationParams((25), Detection.NO_FACE),
            AugmentationParams((45), Detection.NO_FACE)
        ]),
        AddPerChannel([
            AugmentationParams((-55, -35), Detection.FACE),
            AugmentationParams((-35, -15), Detection.FACE),
            AugmentationParams((-10, 10), Detection.FACE),
            AugmentationParams((15, 35), Detection.FACE),
            AugmentationParams((35, 55), Detection.FACE)
       ]),
       AddToHueAndSaturation([
            AugmentationParams((-45), Detection.FACE),
            AugmentationParams((-25), Detection.FACE),
            AugmentationParams((0), Detection.NO_FACE),
            AugmentationParams((25), Detection.NO_FACE),
            AugmentationParams((45), Detection.NO_FACE)
       ]),
       Multiply([
            AugmentationParams((0.25), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1.25), Detection.FACE),
            AugmentationParams((1.5), Detection.FACE)
       ]),
       MultiplyPerChannel([
            AugmentationParams((0.15, 0.35), Detection.FACE),
            AugmentationParams((0.4, 0.6), Detection.FACE),
            AugmentationParams((0.9, 1.1), Detection.FACE),
            AugmentationParams((1.15, 1.35), Detection.FACE),
            AugmentationParams((1.4, 1.6), Detection.FACE)
       ]),
       GaussianBlur([
            AugmentationParams((0.25), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((2.0), Detection.FACE),
            AugmentationParams((4.0), Detection.NO_FACE)
       ]),
       AverageBlur([
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((3), Detection.FACE),
            AugmentationParams((5), Detection.FACE),
            AugmentationParams((7), Detection.FACE),
            AugmentationParams((9), Detection.NO_FACE)
       ]),
       MedianBlur([
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((3), Detection.FACE),
            AugmentationParams((5), Detection.FACE),
            AugmentationParams((7), Detection.FACE),
            AugmentationParams((9), Detection.NO_FACE)
       ]),
       BilateralBlur([
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((3), Detection.FACE),
            AugmentationParams((5), Detection.FACE),
            AugmentationParams((7), Detection.FACE),
            AugmentationParams((9), Detection.FACE)
       ]),
       Sharpen([
            AugmentationParams((0), Detection.NO_FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1.5), Detection.FACE),
            AugmentationParams((2.0), Detection.FACE)
       ]),
       Emboss([
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((1), Detection.FACE),
            AugmentationParams((1.5), Detection.FACE),
            AugmentationParams((2.0), Detection.FACE)
       ]),
       AdditiveGaussianNoise([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.3), Detection.NO_FACE)
       ]),
       AdditiveGaussianNoisePerChannel([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.3), Detection.NO_FACE)
       ]),
       Dropout([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.4), Detection.FACE)
       ]),
       DropoutPerChannel([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.4), Detection.FACE)
       ]),
       CoarseDropout([
            AugmentationParams((0.3), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.1), Detection.NO_FACE),
            AugmentationParams((0.05), Detection.NO_FACE),
            AugmentationParams((0.02), Detection.NO_FACE)
       ]),
       CoarseDropoutPerChannel([
            AugmentationParams((0.3), Detection.NO_FACE),
            AugmentationParams((0.2), Detection.NO_FACE),
            AugmentationParams((0.1), Detection.NO_FACE),
            AugmentationParams((0.05), Detection.NO_FACE),
            AugmentationParams((0.02), Detection.NO_FACE)
       ]),
       SaltAndPepper([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.4), Detection.NO_FACE)
       ]),
       Salt([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.4), Detection.FACE)
       ]),
       Pepper([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.4), Detection.FACE)
       ]),
       CoarseSaltAndPepper([
            AugmentationParams((0.3), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.1), Detection.NO_FACE),
            AugmentationParams((0.05), Detection.NO_FACE),
            AugmentationParams((0.02), Detection.NO_FACE)
       ]),
       CoarseSalt([
            AugmentationParams((0.3), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.1), Detection.NO_FACE),
            AugmentationParams((0.05), Detection.NO_FACE),
            AugmentationParams((0.02), Detection.NO_FACE)
       ]),
       CoarsePepper([
            AugmentationParams((0.3), Detection.FACE),
            AugmentationParams((0.2), Detection.FACE),
            AugmentationParams((0.1), Detection.NO_FACE),
            AugmentationParams((0.05), Detection.NO_FACE),
            AugmentationParams((0.02), Detection.NO_FACE)
       ]),
       ContrastNormalization([
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((0.75), Detection.FACE),
            AugmentationParams((1.0), Detection.FACE),
            AugmentationParams((1.25), Detection.FACE),
            AugmentationParams((1.5), Detection.FACE)
       ]),
       ContrastNormalizationPerChannel([
            AugmentationParams((0.4, 0.6), Detection.FACE),
            AugmentationParams((0.65, 0.85), Detection.FACE),
            AugmentationParams((0.9, 1.1), Detection.FACE),
            AugmentationParams((1.15, 1.35), Detection.FACE),
            AugmentationParams((1.4, 1.6), Detection.FACE)
       ]),
       Grayscale([
            AugmentationParams((0.0), Detection.FACE),
            AugmentationParams((0.25), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((0.75), Detection.FACE),
            AugmentationParams((1.0), Detection.FACE)
       ]),
       PerspectiveTransform([
            AugmentationParams((0.025), Detection.FACE),
            AugmentationParams((0.05), Detection.FACE),
            AugmentationParams((0.075), Detection.FACE),
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.125), Detection.FACE)
       ]),
       PiecewiseAffine([
            AugmentationParams((0.015), Detection.FACE),
            AugmentationParams((0.03), Detection.FACE),
            AugmentationParams((0.045), Detection.FACE),
            AugmentationParams((0.06), Detection.FACE),
            AugmentationParams((0.075), Detection.FACE)
       ]),
       AffineScale([
            AugmentationParams((0.1), Detection.NO_FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((1.0), Detection.FACE),
            AugmentationParams((1.5), Detection.FACE),
            AugmentationParams((1.9), Detection.FACE)
       ]),
       AffineTranslate([
            AugmentationParams((-32, -16), Detection.FACE),
            AugmentationParams((-16, -32), Detection.FACE),
            AugmentationParams((-16, -8), Detection.FACE),
            AugmentationParams((16, 8), Detection.FACE),
            AugmentationParams((16, 32), Detection.FACE)
       ]),
       AffineRotate([
            AugmentationParams((-90), Detection.FACE),
            AugmentationParams((-45), Detection.FACE),
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((45), Detection.FACE),
            AugmentationParams((90), Detection.FACE)
       ]),
       AffineShear([
            AugmentationParams((-45), Detection.FACE),
            AugmentationParams((-25), Detection.FACE),
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((25), Detection.FACE),
            AugmentationParams((45), Detection.FACE)
       ]),
       AffineCval([
            AugmentationParams((0.0), Detection.FACE),
            AugmentationParams((0.25), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((0.75), Detection.FACE),
            AugmentationParams((1.0), Detection.FACE)
       ]),
       ElasticTransformation([
            AugmentationParams((0.1), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((1.0), Detection.FACE),
            AugmentationParams((3.0), Detection.FACE),
            AugmentationParams((9.0), Detection.NO_FACE)
       ]),
       AlphaWithEdgeDetect([
            AugmentationParams((0.0), Detection.FACE),
            AugmentationParams((0.25), Detection.FACE),
            AugmentationParams((0.5), Detection.FACE),
            AugmentationParams((0.75), Detection.FACE),
            AugmentationParams((1.0), Detection.NO_FACE)
       ]),
       AlphaWithEdgeDetectPerChannel([
            AugmentationParams((0.0, 0.2), Detection.FACE),
            AugmentationParams((0.15, 0.35), Detection.FACE),
            AugmentationParams((0.4, 0.6), Detection.FACE),
            AugmentationParams((0.65, 0.85), Detection.FACE),
            AugmentationParams((0.8, 1.0), Detection.NO_FACE)
       ]),
       SimplexNoiseAlphaWithEdgeDetect([
            AugmentationParams((0.0), Detection.FACE),
            AugmentationParams((0.25), Detection.NO_FACE),
            AugmentationParams((0.5), Detection.NO_FACE),
            AugmentationParams((0.75), Detection.FACE),
            AugmentationParams((1.0), Detection.FACE)
       ]),
       FrequencyNoiseAlphaWithEdgeDetect([
            AugmentationParams((-4), Detection.FACE),
            AugmentationParams((-2), Detection.FACE),
            AugmentationParams((0), Detection.FACE),
            AugmentationParams((2), Detection.FACE),
            AugmentationParams((4), Detection.FACE)
       ])
    ]
