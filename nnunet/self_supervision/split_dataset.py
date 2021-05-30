#    Copyright 2021 Division of Medical and Environmental Computing, Technical University of Darmstadt, Darmstadt, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import SimpleITK as sitk


def corrupt_image(filename):
    img_itk = sitk.ReadImage(filename)
    img_npy = sitk.GetArrayFromImage(img_itk)

    dim = img_itk.GetDimension()
    assert dim == 3, "Unexpected dimensionality: %d of file %s, cannot corrupt" % (dim, filename)

    spacing = img_itk.GetSpacing()
    origin = img_itk.GetOrigin()
    direction = np.array(img_itk.GetDirection())

    indices = np.random.random(img_npy.shape) < 0.2
    _img_npy = img_npy.copy()
    _img_npy[indices] = 0  # set to black
    img_itk_new = sitk.GetImageFromArray(_img_npy)
    img_itk_new.SetSpacing(spacing)
    img_itk_new.SetOrigin(origin)
    img_itk_new.SetDirection(direction)

    return img_itk_new


def generate_context_restoration(task_id):
    base = join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')

    task_name = convert_id_to_task_name(task_id)
    target_base = join(base, task_name)

    src = join(target_base, "imagesTr")
    target_ss_input = join(target_base, "ssInputContextRestoration")  # ssInput - corrupted
    target_ss_output = join(target_base, "ssOutputContextRestoration")  # ssOutput - original

    maybe_mkdir_p(target_ss_input)
    maybe_mkdir_p(target_ss_output)

    if isdir(target_ss_output):
        shutil.rmtree(target_ss_output)
    shutil.copytree(src, target_ss_output)

    for file in sorted(listdir(src)):
        corrupt_img = corrupt_image(join(src, file))
        corrupt_img_file = 'corrupted_' + str(file)
        corrupt_img_output = join(target_ss_input, corrupt_img_file)
        sitk.WriteImage(corrupt_img, corrupt_img_output)

    assert len(listdir(target_ss_input)) == len(listdir(target_ss_output)) == len(listdir(src)), \
        "Preparation for self-supervision dataset failed. Check again."


def generate_jigsaw_puzzle():
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="We extend nnUNet to offer self-supervision tasks. This step is to"
                                                 " split the dataset into two - self-supervision input and self- "
                                                 "supervisio output folder.")
    parser.add_argument("-t", type=int, help="Task id. The task name you wish to run self-supervision task for. "
                                             "It must have a matching folder 'TaskXXX_' in the raw "
                                             "data folder")
    parser.add_argument("-ss_tasks", nargs="+", help="Self-supervision Tasks. Specify which self-supervision task you wish to "
                                             "run. Current supported tasks: context restoration (context_resotration)|"
                                             " jigsaw puzzle (jigsaw_puzzle)")
    parser.add_argument("-p", required=False, default=default_num_threads, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    args = parser.parse_args()

    ss_tasks = args.ss_tasks
    if "context_resotration" in ss_tasks:
        generate_context_restoration(args.t)

    if "jigsaw_puzzle" in ss_tasks:
        generate_jigsaw_puzzle()


if __name__ == "__main__":
    main()
