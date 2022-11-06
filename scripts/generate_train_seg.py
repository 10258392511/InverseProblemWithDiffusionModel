import subprocess
import argparse


# The generated bash script should run in submission/


def make_bash_script(hyper_param_dict: dict):
    filename = create_filename(hyper_param_dict)
    bash_script = f"""#!/bin/bash
source $VIRTUALENV_PATH_DIFFUSION
CUDA_VISIBLE_DEVICES=0
cd ../InverseProblemWithDiffusionModel

python3 scripts/train_seg.py --ds_name {hyper_param_dict["ds_name"]} --task_name {hyper_param_dict["task_name"]} --mode {hyper_param_dict["mode"]} --num_workers {hyper_param_dict["num_workers"]}"""

    return bash_script


def create_filename(hyper_param_dict: dict):
    filename = ""
    for key, val in hyper_param_dict.items():
        filename += f"{key}_{val}_"

    filename = filename[:-1].replace(".", "_") + ".sh"

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=[1, 2, 3, 4], required=True)

    args = parser.parse_args()
    hyper_params = dict()
    ds_names = ["ACDC"]
    modes = ["real-valued", "complex"]

    # set 1: ACDC
    for i, ds_name in enumerate(ds_names):
        hyper_params[i + 1] = []
        for mode in modes:
            hyper_params[i + 1].append({
                "ds_name": ds_name,
                "task_name": "Diffusion",
                "mode": mode,
                "num_workers": 0
            })

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
