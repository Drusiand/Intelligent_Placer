from intelligent_placer_lib import intelligent_placer
import sys

ERROR_TOKEN = 'ERROR: '
WRONG_CONFIG = 'Wrong launch configuration, consider using --path or -p for passing image path as argument'
UNKNOWN_PARAM = 'Unknown parameter format'

PATH_PARAM_NAMES = {'--path', '-p'}


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(ERROR_TOKEN + WRONG_CONFIG)
        exit(1)
    else:
        param_name = sys.argv[1]
        path_arg = sys.argv[2]
        if param_name in PATH_PARAM_NAMES:
            intelligent_placer.check_image(path_arg)
        else:
            print(ERROR_TOKEN + UNKNOWN_PARAM)
            exit(2)
