from intelligent_placer_lib import intelligent_placer
import sys

ERROR_TOKEN = 'ERROR: '
WRONG_CONFIG = 'Wrong launch configuration, consider using --path or -p for passing image path as argument'
UNKNOWN_PARAM = 'Unknown parameter format'

PATH_PARAM_NAMES = {'--path', '-p'}


def test_intelligent_placer(path):
    assert intelligent_placer.check_image(path) == []
    print('test ' + path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(ERROR_TOKEN + WRONG_CONFIG)
    else:
        param_name = sys.argv[1]
        path_arg = sys.argv[2]
        if param_name in PATH_PARAM_NAMES:
            test_intelligent_placer(path_arg)
        else:
            print(ERROR_TOKEN + UNKNOWN_PARAM)
