import time
import os
import argparse

import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import openslide
from tqdm import tqdm

import numpy as np

from clam_utils.file_utils import save_hdf5
from augment_utils.transform_utils import get_random_transforms, get_standard_transforms, IdentityTransform
from models.wsi_dataset import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.model_builder import get_encoder

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Embedder Device: ", device)

def viz_subbatch(batch_tensor):
	import matplotlib.pyplot as plt
	images = batch_tensor.cpu().movedim(1,3).numpy()
	fig, axs = plt.subplots(1, 5,figsize=(15,8))
	for i in range(5):
		axs[i].imshow(images[i])
		#axs[i].set_title("Target Patch")
	plt.show()


def compute_feats_one_slide(output_path, loader, model, verbose = 0):
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(loader):
		t1=time.time()
		with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
			#doubly batched by default collate (unless batch size is 1)
			batch = data['img'].squeeze()
			#viz_subbatch(batch)
			coords = data['coord'].squeeze(0).numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			try:
				out = model(batch)
				#thanks virchow
				features = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1) if model.name=='virchow' else out
			except Exception as e:
				print(e)
				print("Embedder failed on batch, likely batch size of 1, ignored")
				print("Batch Size", batch.shape)
				continue
			features = features.detach().cpu().numpy().astype(np.float32)
			asset_dict = {'features': features, 'coords': coords} #, 'pos_x':coords[:,0],'pos_y':coords[:,1]}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
			#print(f"Embedder needs {time.time()-t1}s for 1 batch\n")
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--feat_dir', type=str, default='/home/millz/project_data/histo_data/feats_256')
parser.add_argument('--data_patches_dir', type=str, default='/home/millz/project_data/histo_data/clam_256')
parser.add_argument('--data_slide_dir', type=str, default='/media/millz/T7 Shield/data/histo_wsi')
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--model_name', type=str, default='uni_v1', choices=['uni_v1', 'conch_v1','jingsong_fm', 'gigapath', 'chief', 'virchow'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--augment_method', type=str, default='no_augment',choices=['no_augment','rc-pure','rc-mix','rc-up','macenko-norm','macenko-aug'])
parser.add_argument('--batch_randomize', default=False, action='store_true',help='whether to randomize augmenter before each batch (default: before every new slide)')
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError("Must supply a csv of slides to process")

	bags_dataset = Dataset_All_Bags(csv_path)
	#no speedup observed when using second cuda device
	loader_device = torch.device('cpu')#torch.device('cuda:0') #device
	#TODO: change back to 8
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if loader_device.type == "cpu" else {}
	print("Data Loading and Transforming Settings: ",loader_device)
	print(loader_kwargs)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'h5_files'))

	encoder = get_encoder(args.model_name)
	if(args.augment_method=='no_augment'):
		img_transforms = get_standard_transforms(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		augmenter_pointer = IdentityTransform()
	else:
		img_transforms, augmenter_pointer = get_random_transforms(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],method=args.augment_method,batch_randomize=args.batch_randomize,device=loader_device)
	_ = encoder.eval()
	encoder = encoder.to(device)
	total = len(bags_dataset)

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)
		bag_name = slide_id+'.h5'
		if not args.no_auto_skip and bag_name in dest_files:
			print('skipped {}'.format(slide_id))
			continue 
		h5_file_path = os.path.join(args.data_patches_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		#randomize the augments for each WSI 
		augmenter_pointer.reset(slide_id)
		try:
			patch_dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
		except FileNotFoundError as e:
			print(f"No such file: {h5_file_path}\n Redo patching")
			continue
		#the batch sampler ensures that the whole batch goes through the transform at once, rather than one by one and collated after
		batch_sampler = BatchSampler(SequentialSampler(patch_dataset),batch_size=args.batch_size, drop_last=False)
		#pass as sampler to get batch of idxs passed, pass as batch_sampler to do 1by1 and collate
		patch_loader = DataLoader(dataset=patch_dataset,sampler=batch_sampler,**loader_kwargs)
		output_file_path = compute_feats_one_slide(output_path, loader = patch_loader, model = encoder, verbose = 1)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
	#augmenter_pointer.augmenter.save_cache('stain_matrices/vahadane-cache_tcga.pkl')


