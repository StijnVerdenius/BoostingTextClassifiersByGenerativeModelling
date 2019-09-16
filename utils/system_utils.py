from utils.constants import *


def ensure_current_directory():
    """
        ensures we run from main directory even when we run testruns
        :return:
        """

    current_dir = os.getcwd()
    base_name = PROJ_NAME
    os.chdir(os.path.join(current_dir.split(base_name)[0], base_name))


def setup_directories():
    stamp = DATA_MANAGER.stamp
    dirs = OUTPUT_DIRS
    for dir_to_be in dirs:
        DATA_MANAGER.create_dir(os.path.join(RESULTS_DIR, stamp, dir_to_be))


def save_codebase_of_run(arguments):
    directory = os.path.join(GITIGNORED_DIR, RESULTS_DIR, DATA_MANAGER.stamp, CODEBASE_DIR)
    f = open(os.path.join(directory, "arguments.txt"), "w")
    f.write(str(arguments).replace(", ", "\n"))
    f.close()

    stack = ["."]

    while len(stack) > 0:

        path = stack.pop(0)

        for file_name in os.listdir(os.path.join(os.getcwd(), path)):

            if file_name.endswith(".py"):
                f = open(str(os.path.join(directory, file_name)).replace(".py", ""), "w")
                lines = open(str(os.path.join(path, file_name)), "r").read()
                f.write(str(lines))
                f.close()
            elif os.path.isdir(os.path.join(os.getcwd(), path, file_name)):
                stack.append(os.path.join(path, file_name))

    base = os.path.join(os.getcwd(), GITIGNORED_DIR, RESULTS_DIR, DATA_MANAGER.stamp, CODEBASE_DIR)
    for file_name in list(os.listdir(base)):
        if ("arguments.txt" in file_name): continue
        os.rename(base + "/" + file_name, base + "/" + file_name + ".py")
