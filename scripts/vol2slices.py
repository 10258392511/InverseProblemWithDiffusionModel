import sys
import os

# path = "/scratch/zhexwu"
path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not path in sys.path:
    sys.path.append(path)


from InverseProblemWithDiffusionModel.helpers.load_data import vol2slice

# change this
root_dir = "/scratch/shared/ACDC_textures/data"
save_dir = "/scratch/zhexwu/data/ACDC_textures/data_slices"


if __name__ == '__main__':
    vol2slice(root_dir, save_dir)
