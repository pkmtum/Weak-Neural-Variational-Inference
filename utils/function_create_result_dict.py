def create_result_dict(dir_name=None):
    """
    This function creates a directory to save the plots.
    """
    # set default directory to store plots
    from datetime import datetime
    import os

    # get root
    cwd = os.getcwd()
    root_path = os.path.join(cwd, "results")
    assert os.path.isdir(root_path), f"There is no directory with result path {root_path}"

    # result folder name is Date_#_i
    folder_name_root = datetime.today().strftime('%Y_%m_%d')
    # make a custom name for the dir
    if dir_name is not None:
        folder_name_root = folder_name_root + "_" + dir_name

    # gets i for new directory Date_#_i
    # just for iteration 0 ...
    results_path = root_path
    i = 0
    while os.path.isdir(results_path):
        folder_name = folder_name_root + "_#_" + str(int(i))
        results_path = os.path.join(root_path, folder_name)
        i += 1

    # create the result dir
    os.mkdir(results_path)
    print(f"Created results directory: {results_path}.")

    # return path
    return results_path
