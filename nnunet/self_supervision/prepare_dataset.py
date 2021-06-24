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
from nnunet.preprocessing.sanity_checks import verify_same_geometry, verify_all_same_orientation
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

    print(f"Generated dataset for {task_name}.")
    return file_names


def verify_dataset_integrity(folder, ss_task):
    """
    folder needs the ssInputTASK and ssOutputTASK sub-folder. There also needs to be a dataset.json
    checks if all input and output images are present
    for each case, checks whether all modalities are present
    for each case, checks whether the pixel grids are aligned
    :param folder:
    :return:
    """
    assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
    dataset = load_json(join(folder, "dataset.json"))

    # check if the pretext task has "ssInputTASK" and "ssOutputTASK"
    pretext_task = convert_pretext_task(ss_task)
    input_folder = f"ssInput{pretext_task}"
    output_folder = f"ssOutput{pretext_task}"

    assert isdir(join(folder, input_folder)), \
        f"There needs to be a {input_folder} subfolder in folder, folder={folder}"
    assert isdir(join(folder, output_folder)), \
        f"There needs to be a {output_folder} subfolder in folder, folder={folder}"

    training_cases = dataset[pretext_task.lower()]
    num_modalities = len(dataset['modality'].keys())
    expected_train_identifiers = [i['input'].split("/")[-1][:-12] for i in training_cases]
    expected_test_identifiers = [i['output'].split("/")[-1][:-12] for i in training_cases]

    ## check training set
    nii_files_in_pretext_input = subfiles((join(folder, input_folder)), suffix=".nii.gz", join=False)
    nii_files_in_pretext_output = subfiles((join(folder, output_folder)), suffix=".nii.gz", join=False)
    nii_files_in_labelsTr = subfiles((join(folder, "labelsTr")), suffix=".nii.gz", join=False)

    label_files = []
    geometries_OK = True
    has_nan = False

    # check all cases
    if len(expected_train_identifiers) != len(np.unique(expected_train_identifiers)):
        raise RuntimeError("found duplicate input cases in dataset.json")
    if len(expected_train_identifiers) != len(expected_test_identifiers):
        raise RuntimeError("input and output cases in dataset.json doesn't match")

    print("Verifying SS output set") # same images in ImageTr
    for c in expected_test_identifiers:
        print("checking case", c)
        # check if all files are present
        expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
        label_files.append(expected_label_file)

        expected_output_files = [join(folder, output_folder, c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]

        assert all([isfile(i) for i in
                    expected_output_files]), "some image files are missing for case %s. Expected files:\n %s" % (
            c, expected_output_files)

        # verify that all modalities and the label have the same shape and geometry.
        label_itk = sitk.ReadImage(expected_label_file)

        inputs_itk = [sitk.ReadImage(i) for i in expected_output_files]
        for i, img in enumerate(inputs_itk):
            nans_in_image = np.any(np.isnan(sitk.GetArrayFromImage(img)))
            has_nan = has_nan | nans_in_image
            same_geometry = verify_same_geometry(img, label_itk)
            if not same_geometry:
                geometries_OK = False
                print(
                    "The geometry of the image %s does not match the geometry of the label file. The pixel arrays "
                    "will not be aligned and nnU-Net cannot use this data. Please make sure your image modalities "
                    "are coregistered and have the same geometry as the label" % expected_output_files[0][:-12])
            if nans_in_image:
                print("There are NAN values in image %s" % expected_output_files[i])

        # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
        for i in expected_output_files:
            nii_files_in_pretext_output.remove(os.path.basename(i))
        nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

    # check for stragglers
    assert len(nii_files_in_pretext_output) == 0, f"there are training cases in {output_folder} that are not listed in " \
                                                 f"dataset.json: %s" % nii_files_in_pretext_output
    assert len(nii_files_in_labelsTr) == 0, "there are training cases in labelsTr that are not listed in " \
                                            "dataset.json: %s" % nii_files_in_labelsTr

    # check SS input set
    if len(expected_train_identifiers) > 0:
        print("Verifying input set")

        for c in expected_train_identifiers:
            print("checking case", c)
            # check if all files are present
            expected_input_files = [join(folder, input_folder, c + "_%04.0d.nii.gz" % i) for i in
                                     range(num_modalities)]
            assert all([isfile(i) for i in
                        expected_input_files]), "some image files are missing for case %s. Expected files:\n %s" % (
                c, expected_input_files)

            # verify that all modalities and the label have the same geometry. We use the affine for this
            if num_modalities > 1:
                images_itk = [sitk.ReadImage(i) for i in expected_input_files]
                reference_img = images_itk[0]

                for i, img in enumerate(images_itk[1:]):
                    assert verify_same_geometry(img, reference_img), \
                        "The modalities of the image %s do not seem to be registered. "\
                        "Please coregister your modalities." % (expected_input_files[i])

            # now remove checked files from the lists nii_files_in_pretext_output
            for i in expected_input_files:
                nii_files_in_pretext_input.remove(os.path.basename(i))
        assert len(nii_files_in_pretext_input) == 0, f"there are training cases in {input_folder} that are not " \
                                                      "listed in dataset.json: %s" % nii_files_in_pretext_input

    all_same, unique_orientations = verify_all_same_orientation(join(folder, output_folder))
    if not all_same:
        print(
            "WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you "
            "correct that by reorienting the data. fslreorient2std should do the trick")

    # save unique orientations to dataset.json
    if not geometries_OK:
        raise Warning(
            "GEOMETRY MISMATCH FOUND! CHECK THE TEXT OUTPUT! This does not cause an error at this point but you "
            "should definitely check whether your geometries are alright!")
    else:
        print("Self Supervision Dataset OK")

    if has_nan:
        raise RuntimeError(
            "Some images have nan values in them. This will break the training. See text output above to see which ones")


def convert_pretext_task(pretext_task):
    """ key: arg command; value: file name """
    pretext_tasks = {
        'context_restoration': 'ContextRestoration',
        'jigsaw_puzzle': 'JigsawPuzzle',
        'byol': 'BYOL',
    }

    return pretext_tasks[pretext_task]


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
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    args = parser.parse_args()

    ss_tasks = args.ss_tasks
    base = join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')
    task_name = convert_id_to_task_name(args.t)
    target_base = join(base, task_name)

    import json
    with open(join(target_base, 'dataset.json')) as json_file:
        updated_json_file = json.load(json_file).copy()

    if "context_restoration" in ss_tasks:
        pretext_task = convert_pretext_task("context_restoration")
        file_names = generate_augmented_datasets(pretext_task, target_base, swap_image)
        updated_json_file['contextRestoration'] = [{'input': f"./ssInput{pretext_task}/_%s" % i.split("/")[-1],
                                                    "output": f"./ssOutput{pretext_task}/%s" % i.split("/")[-1]} \
                                                   for i in file_names]
        if args.verify_dataset_integrity:
            verify_dataset_integrity(target_base, "context_restoration")

    if "jigsaw_puzzle" in ss_tasks:
        pretext_task = convert_pretext_task("jigsaw_puzzle")
        file_names = generate_augmented_datasets(pretext_task, target_base, swap_image)
        updated_json_file['jigsawPuzzle'] = [{'input': f"./ssInput{pretext_task}/_%s" % i.split("/")[-1],
                                              "output": f"./ssOutput{pretext_task}/%s" % i.split("/")[-1]} \
                                             for i in file_names]
        if args.verify_dataset_integrity:
            verify_dataset_integrity(target_base, "jigsaw_puzzle")

    if "byol" in ss_tasks:
        pretext_task = convert_pretext_task("byol")
        file_names = generate_augmented_datasets(pretext_task, target_base, byol_aug)
        updated_json_file['byol'] = [{'input': f"./ssInput{pretext_task}/_%s" % i.split("/")[-1],
                                      "output": f"./ssOutput{pretext_task}/%s" % i.split("/")[-1]} \
                                     for i in file_names]
        if args.verify_dataset_integrity:
            verify_dataset_integrity(target_base, "byol")

    # remove the original dataset.json
    os.remove(join(target_base, 'dataset.json'))
    # remove the modified dataset.json
    save_json(updated_json_file, join(target_base, "dataset.json"))
    print('Updated dataset.json')

    # run sanity check for input/output images

    print('Preparation for self supervision succeeded! Move on to the plan_and_preprocessing stage.')


if __name__ == "__main__":
    main()
