import errno
import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
import SimpleITK as sitk
from multiprocessing import Pool


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
    _img_npy[indices] = 0 # set to black
    img_itk_new = sitk.GetImageFromArray(_img_npy)
    img_itk_new.SetSpacing(spacing)
    img_itk_new.SetOrigin(origin)
    img_itk_new.SetDirection(direction)

    return img_itk_new
    # sitk.WriteImage(img_itk_new, f'corrupted_{filename}')


def main():
    # local file path for testing
    base = '/Users/juliefang/Documents/nnUNet_raw_data_base/nnUNet_raw_data/nnUNet_raw_data'

    task_name = 'Task002_Heart'
    target_base = join(base, task_name)

    target_ss_input = join(target_base, "ssInput")  # ssInputContextRestoration - corrupted
    target_ss_output = join(target_base, "ssOutput")  # ssOutputContextRestoration - original

    maybe_mkdir_p(target_ss_input)
    maybe_mkdir_p(target_ss_output)

    src = join(target_base, "imagesTr")
    dest = join(target_base, "ssOutput")

    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)
            shutil.copytree(src, dest)
    except OSError as e:
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Error occurs: ' + str(e))

    print(listdir(dest))

    dest = join(target_base, 'ssInput')

    for file in sorted(listdir(src)):
        corrupt_img = corrupt_image(join(src,file))
        corrupt_img_file = 'c' + str(file)
        corrupt_img_output = join(dest, corrupt_img_file)
        sitk.WriteImage(corrupt_img, corrupt_img_output)

    if listdir(target_ss_input) == listdir(target_ss_output):
        print("Copied success: self-supervision dataset is ready.")

if __name__ == "__main__":
    main()
