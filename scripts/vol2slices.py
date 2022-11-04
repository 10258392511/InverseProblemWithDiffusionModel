from InverseProblemWithDiffusionModel.helpers.load_data import vol2slice

# change this
root_dir = "E:\Datasets\ACDC_textures\data"
save_dir = "E:\Datasets\ACDC_textures\data_slices"


if __name__ == '__main__':
    vol2slice(root_dir, save_dir)
