import numpy as np
import torch
from torchvision.transforms import v2

from augment_utils.augment_utils import StainAugment, RandomConvAugment

from time import time

def get_standard_transforms(mean, std):
	trsforms =  v2.Compose([
	v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True), 	#already uint8, but just to be sure
	#v2.Resize((256,256),interpolation=v2.InterpolationMode.BILINEAR),
	v2.Resize((256,256),interpolation=v2.InterpolationMode.BICUBIC),
	v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
	lambda x: torch.stack(x),
	v2.Normalize(mean, std)
	])

	return trsforms


def get_random_transforms(mean,std,method,batch_randomize,device):
	custom_augmenter = ConsistentStainTransform(method,batch_randomize,device)
	trsforms =  v2.Compose([
	v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image, possibly also .ToPureTensor
	lambda x: torch.stack(x).to(device),
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    #v2.Resize((256,256),interpolation=v2.InterpolationMode.BILINEAR),
	v2.Resize((256,256),interpolation=v2.InterpolationMode.BICUBIC),
	v2.RandomCrop(224),
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
	custom_augmenter,
    v2.Normalize(mean=mean, std=std,inplace=True)
	])
	return trsforms, custom_augmenter
			
def get_patch_qc_transform():
	trsforms=v2.Compose([
		v2.ToImage(),
		v2.Resize(256),
		v2.CenterCrop(96),
	])
	return trsforms

class ConsistentStainTransform(object):
	def __init__(self, method, batch_randomize, device=None):
		self.method=method
		self.device = device
		self.jit_factor=1
		self.batch_randomize=batch_randomize
		self.rng = np.random.default_rng()
		print(f"Setting Up Transform of Type {method}")
		if(method in ['vahadane-norm','macenko-norm']):
			self.augmenter = StainAugment(method)
			self.jitter = None
			self.rot=0
		elif(method in ['vahadane-aug','macenko-aug']):
			self.augmenter = StainAugment(method)
			self.jitter = None
			self.rot=180
		elif(method=='rc-mix'):
			self.augmenter = RandomConvAugment(3,7,1.0,0.5)
			self.jitter = lambda x,y: x
			self.rot=90
		elif(method=='rc-pure'):
			self.augmenter = RandomConvAugment(num_layers=3,kernel_size=3,prob=1.0,mix_weight=1.0)
			self.jitter = lambda x,y: x
			self.rot=None
		else:
			raise NotImplementedError
		
		self.augmenter.to(self.device)

	def __call__(self, image):
		if(self.batch_randomize): self.augmenter.randomize()
		image=v2.functional.horizontal_flip(image) if self.method=='rc-pure' else v2.functional.rotate(image,self.rot)
		try:
			image = self.augmenter(image)
		except Exception as e:
			print("Transform Failed. Likely batch from low quality/artifact image region. Ignoring batch.")
			print(e)
			#very hacky way of returning a dummy vec which will then later be ignored 
			return torch.ones((1,3,224,224))
		#image=self.jitter(image,self.jit_factor)
		return image

	def reset(self,slide_id):
		self.augmenter.randomize()
		#self.jit_factor=self.rng.uniform(0.75,1.3)



class IdentityTransform(object):
	def __init__(self):
		super(IdentityTransform, self).__init__()

	def __call__(self, image):
		return image
	
	def reset(self,slide_id):
		return