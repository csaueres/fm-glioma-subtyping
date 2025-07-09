import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from PIL import Image
import h5py
from time import time

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		#self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def get_patching_attrs(self):
		patch_attrs = {'patch_size':self.patch_size,'patch_level':self.patch_level}
		return patch_attrs

	def __getitem__(self, idx):
			with h5py.File(self.file_path,'r') as hdf5_file:
				coords = hdf5_file['coords'][idx]
			if isinstance(idx,list):
				imgs=[]
				patch_ids=[]
				for c in coords:
					#c=coords[i]
					img = self.wsi.read_region(c, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
					imgs.append(img)
			else:
				imgs=self.wsi.read_region(coords, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
				patch_ids=self.file_path+'_'+str(idx)
			imgs = self.roi_transforms(imgs)
			return {'img': imgs, 'coord':coords}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path,dtype={'slide_id': str})
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




