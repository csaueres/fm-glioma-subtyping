import time
import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import openslide
from tqdm import tqdm


from clam_utils.file_utils import save_hdf5
from augment_utils.transform_utils import IdentityTransform, get_patch_qc_transform
from models.wsi_dataset import Dataset_All_Bags, Whole_Slide_Bag_FP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", device)

def viz_subbatch(patch_batch):
    import matplotlib.pyplot as plt
    n_pics = min(len(patch_batch),5)
    if(n_pics<=1):
          return
    images = np.moveaxis(patch_batch,1,3)
    fig, axs = plt.subplots(1, n_pics,figsize=(15,8))
    for i in range(n_pics):
        axs[i].imshow(images[i])
    plt.show()

def check_batch_for_faulty_patch(batch):
    patch_size = batch.shape[2]
    cc = patch_size//2
    patch_sd = np.std(batch,(2,3))
    # print(patch_sd[:64])
    patch_sd_mean = np.mean(patch_sd,1)
    return patch_sd_mean<10
	

def process_patches_one_slide(output_path, loader):
    n_patches_removed=0
    mode = 'w'
    attrs = loader.dataset.get_patching_attrs()
    for count, data in enumerate(loader):
        patches = data['img'].numpy()
        coords = data['coord'].numpy().astype(np.int32)
        faulty_patch_mask = check_batch_for_faulty_patch(patches)
        try:
            asset_dict = {'coords': coords[~faulty_patch_mask]}
            save_hdf5(output_path, asset_dict, attr_dict= {'coords':attrs}, mode=mode)
            n_patches_removed+=sum(faulty_patch_mask)
            mode = 'a'
        except Exception as e:
              print("Error when saving cleaned preds. Likely all patches axed")
              print(asset_dict)
              #skip this batch
              continue

    return n_patches_removed


parser = argparse.ArgumentParser(description='Patch Filtration')
parser.add_argument('--out_patches', type=str, default='/home/millz/project_data/histo_data/clam_256_corr')
parser.add_argument('--in_patches', type=str, default='/home/millz/project_data/histo_data/clam_256')
parser.add_argument('--data_slide_dir', type=str, default='/media/millz/T7 Shield/data/histo_wsi')
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError("Must supply a csv of slides to process")

	bags_dataset = Dataset_All_Bags(csv_path)
	loader_device = torch.device('cpu') #device
	loader_kwargs = {'num_workers': 4, 'pin_memory': False} if loader_device.type == "cpu" else {}
	
	os.makedirs(args.out_patches, exist_ok=True)
	os.makedirs(os.path.join(args.out_patches, 'patches'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.out_patches, 'patches'))

	total = len(bags_dataset)

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)
		bag_name = slide_id+'.h5'
		if not args.no_auto_skip and bag_name in dest_files:
			print('skipped {}'.format(slide_id))
			continue 
		h5_file_path = os.path.join(args.in_patches, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		output_path = os.path.join(args.out_patches, 'patches', bag_name)

		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		patch_dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=get_patch_qc_transform())
		patch_loader = DataLoader(dataset=patch_dataset,batch_size=args.batch_size,**loader_kwargs)
		n_patches_removed = process_patches_one_slide(output_path, loader = patch_loader)
		time_elapsed = time.time() - time_start
		print('\nprocessing {} took {}s. {} patches of {} were removed'.format(output_path, np.round(time_elapsed,2), n_patches_removed,len(patch_dataset)))