import shutil

def copy_input_file(input_file_path, results_path):
    """
    This function copies the input file to the results directory.
    """
    # copy input file to results directory
    shutil.copy(input_file_path, results_path)