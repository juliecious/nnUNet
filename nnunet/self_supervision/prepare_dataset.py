######################################################################################
# ----------Copyright 2021 Division of Medical and Environmental Computing,----------#
# ----------Technical University of Darmstadt, Darmstadt, Germany--------------------#
######################################################################################

import shutil

import SimpleITK as sitk
import numpy as np
import torchio as tio
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.configuration import default_num_threads
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def get_sitk_data(filename):
    """ Return meta information of the given NIfTI image """
    img_itk = sitk.ReadImage(filename)
    img_npy = sitk.GetArrayFromImage(img_itk)

    dim = img_itk.GetDimension()
    assert dim == 3, "Unexpected dimensionality: %d of file %s, cannot corrupt" % (dim, filename)

    spacing = img_itk.GetSpacing()
    origin = img_itk.GetOrigin()
    direction = np.array(img_itk.GetDirection())

    return img_itk, img_npy, spacing, origin, direction


def add_noise_to_image(filename):
    """ Image manipulation method - add noise """
    image = tio.ScalarImage(filename)
    noise = tio.RandomNoise()
    noised = noise(image)
    return noised


def swap_image(filename):
    """ Image manipulation method - swap patches of image for context restoration """
    image = tio.ScalarImage(filename)
    swap = tio.RandomSwap()
    swapped = swap(image)
    return swapped


def byol_aug(filename):
    """
        BYOL minimizes the distance between representations of each sample and a transformation of that sample.
        Examples of transformations include: translation, rotation, blurring, color inversion, color jitter, gaussian noise.

        Return an augmented dataset that consisted the above mentioned transformation. Will be used in the training.
        """
    image = tio.ScalarImage(filename)
    get_foreground = tio.ZNormalization.mean
    training_transform = tio.Compose([
        tio.CropOrPad((180, 220, 170)),  # zero mean, unit variance of foreground
        tio.ZNormalization(
            masking_method=get_foreground),
        tio.RandomBlur(p=0.25),  # blur 25% of times
        tio.RandomNoise(p=0.25),  # Gaussian noise 25% of times
        tio.OneOf({  # either
            tio.RandomAffine(): 0.8,  # random affine
            tio.RandomElasticDeformation(): 0.2,  # or random elastic deformation
        }, p=0.8),  # applied to 80% of images
        tio.RandomBiasField(p=0.3),  # magnetic field inhomogeneity 30% of times
        tio.OneOf({  # either
            tio.RandomMotion(): 1,  # random motion artifact
            tio.RandomSpike(): 2,  # or spikes
            tio.RandomGhosting(): 2,  # or ghosts
        }, p=0.5),  # applied to 50% of images
    ])

    tfs_image = training_transform(image)
    return tfs_image


def generate_augmented_datasets(task_name, target_base, aug_fn):
    src = join(target_base, "imagesTr")
    target_ss_input = join(target_base, "ssInput" + task_name)  # ssInput - pretext tasks
    target_ss_output = join(target_base, "ssOutput" + task_name)  # ssOutput - original images copied from ImagesTr

    maybe_mkdir_p(target_ss_input)
    maybe_mkdir_p(target_ss_output)

    # copy all files in ImagesTr to ssOutput folder
    if isdir(target_ss_output):
        shutil.rmtree(target_ss_output)
    shutil.copytree(src, target_ss_output)

    file_names = []
    # copy augmented images to ssInput folder
    for file in sorted(listdir(src)):
        corrupt_img = aug_fn(join(src, file))
        corrupt_img_file = "_" + str(file)
        file_names.append(str(file))
        corrupt_img_output = join(target_ss_input, corrupt_img_file)
        corrupt_img.save(corrupt_img_output)

    # sanity check
    assert len(listdir(target_ss_input)) == len(listdir(target_ss_output)) == len(listdir(src)), \
        f"Self-supervision dataset generation for {task_name} failed. Check again."

    return file_names


def main():
    import argparse
    parser = argparse.ArgumentParser(description="We extend nnUNet to offer self-supervision tasks. This step is to"
                                                 " split the dataset into two - self-supervision input and self- "
                                                 "supervision output folder.")
    parser.add_argument("-t", type=int, help="Task id. The task name you wish to run self-supervision task for. "
                                             "It must have a matching folder 'TaskXXX_' in the raw "
                                             "data folder", required=True)
    parser.add_argument("-ss_tasks", nargs="+",
                        help="Self-supervision Tasks. Specify which self-supervision task you wish to "
                             "run. Current supported tasks: context_restoration| jigsaw_puzzle | byol")
    parser.add_argument("-p", default=default_num_threads, type=int, required=False,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    args = parser.parse_args()

    ss_tasks = args.ss_tasks
    base = join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')
    task_name = convert_id_to_task_name(args.t)
    target_base = join(base, task_name)

    import json
    with open(join(target_base, 'dataset.json')) as json_file:
        updated_json_file = json.load(json_file).copy()

    if "context_restoration" in ss_tasks:
        file_names = generate_augmented_datasets("ContextRestoration", target_base, swap_image)
        updated_json_file['contextRestoration'] = [{'image': "./ssInputContextRestoration/%s.nii.gz" % i.split("/")[-1], \
                                                    "label": "./ssOutputContextRestoration/%s.nii.gz" % i.split("/")[
                                                        -1]} for i in file_names]
        print('Prepared dataset for context restoration.')

    if "jigsaw_puzzle" in ss_tasks:
        file_names = generate_augmented_datasets("JigsawPuzzle", target_base, swap_image)
        updated_json_file['jigsawPuzzle'] = [{'image': "./ssInputJigsawPuzzle/%s.nii.gz" % i.split("/")[-1], \
                                              "label": "./ssOutputJigsawPuzzle/%s.nii.gz" % i.split("/")[
                                                  -1]} for i in file_names]
        print('Prepared dataset for jigsaw puzzle.')

    if "byol" in ss_tasks:
        file_names = generate_augmented_datasets("BYOL", target_base, byol_aug)
        updated_json_file['byol'] = [{'image': "./ssInputBYOL/%s.nii.gz" % i.split("/")[-1], \
                                      "label": "./ssOutputBYOL/%s.nii.gz" % i.split("/")[
                                          -1]} for i in file_names]
        print('Prepared dataset for byol.')

    # remove the original dataset.json
    os.remove(join(target_base, 'dataset.json'))
    # remove the modified dataset.json
    save_json(updated_json_file, join(target_base, "dataset.json"))
    print('Updated dataset.json')

    print('Preparation for self supervision task succeeded! Move on to the plan_and_preprocessing stage.')


if __name__ == "__main__":
    main()
