import argparse
import pkg_resources
from utils.load_params import load_params

def write_pkg_list_file(pkg_list_fname):
    packages = pkg_resources.working_set
    packages_list = sorted([f"{i.key}=={i.version}" for i in packages])
    with open(pkg_list_fname, 'w') as f:
        for item in packages_list:
            f.write(f"{item}\n")



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = load_params(params_path=args.config)
    pkg_list_fname = params.base.pkg_list_fname
    write_pkg_list_file(pkg_list_fname=pkg_list_fname)