######################################################################################
# ----------Copyright 2021 Division of Medical and Environmental Computing,----------#
# ----------Technical University of Darmstadt, Darmstadt, Germany--------------------#
######################################################################################

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


# Main driver for running self supervised learning pretext tasks
def main():
    import argparse
    parser = argparse.ArgumentParser(description="We extend nnUNet to offer self-supervision tasks. This step is to"
                                                 " split the dataset into two - self-supervision input and self- "
                                                 "supervisio output folder.")
    parser.add_argument("-t", type=int, help="Task id. The task name you wish to run self-supervision task for. "
                                             "It must have a matching folder 'TaskXXX_' in the raw "
                                             "data folder", required=True)
    parser.add_argument("-ss", help="Run self-supervision pretext asks. Specify which self-supervision task you "
                             "wish to train. Current supported tasks: context_restoration| jigsaw_puzzle | byol")

    args = parser.parse_args()

    base = join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')
    task_name = convert_id_to_task_name(args.t)
    target_base = join(base, task_name)
    pretext = str(args.ss)

    print(f'Hey there: here\'s pretext task {pretext} for {task_name}. '
          f'Path to get ss datasets are {join(target_base, "ssInput" + "BYOL")} and {join(target_base, "ssOutput" + "BYOL")}')


if __name__ == "__main__":
    main()
