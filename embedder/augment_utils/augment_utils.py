import numpy as np
from torch import nn, load, mean
from torch_staintools.normalizer import NormalizerBuilder
from torch_staintools.augmentor import AugmentorBuilder
from time import time


class RandomConvAugment(object):
	def __init__(self, num_layers,kernel_size,prob,mix_weight):
		self.n_layer=num_layers
		self.k=kernel_size
		self.prob = prob
		#compute a weighted sum with the original image, where the weight is for the distortion
		self.mix_weight=mix_weight
		assert mix_weight<=1.0 and mix_weight>0.0
		self.random_conv = nn.Conv2d(3, 3, kernel_size=self.k, padding=self.k//2,bias=False)
		self.re_init_weights()
		for param in self.random_conv.parameters():
			param.requires_grad = False
		self.act=nn.Tanh()
		self.rng = np.random.default_rng()

	def __call__(self, image):
		return self.transform(image)
	
	def transform(self,image):
		conv_image = self.random_conv(image)
		#conv_image = self.act(conv_image)
		for i in range(self.n_layer-1):
			conv_image = self.random_conv(conv_image)
			conv_image = self.act(conv_image)
		conv_image = conv_image-conv_image.min()
		conv_image = conv_image/conv_image.max()
		return self.mix_weight*conv_image+(1-self.mix_weight)*image
	
	def to(self,dev):
		self.random_conv = self.random_conv.to(dev)
		
	def re_init_weights(self):
		nn.init.normal_(self.random_conv.weight,0, 1. / (self.k * self.k))
	
	#pseudonym to be compatible with HED augmenter
	def randomize(self):
		self.re_init_weights()


#just a wrapper for the stain normalization/augmentation classes so they are compatible with the other augmenters
class StainAugment(object):
	def __init__(self,method):
		self.method=method
		self.target_tensor = self._get_target_tensor()
		if(method=='macenko-norm'):
			#self.cache_fp="./macenko-cache.pkl"
			self.normalizer = NormalizerBuilder.build('macenko',concentration_method='ls',use_cache=False,load_path=None)
			self.normalizer.fit(self.target_tensor)
		elif(method=='macenko-aug'):
			#self.cache_fp="./macenko-cache.pkl"
			self.normalizer = AugmentorBuilder.build('macenko',
                                   rng=7,
                                   luminosity_threshold=0.8,
                                   concentration_method='ls',
                                   sigma_alpha=0.2, sigma_beta=0.2, target_stain_idx=(0, 1),
                                   use_cache=False,
                                   cache_size_limit=-1,
                                   # if specified, the augmentor will load the cached stain matrices from file system.
                                   load_path=None,
                                   )
		else:
			raise NotImplementedError
		
		print(self.normalizer.stain_matrix_target)
		

	def __call__(self, image):
		mean_intensities = mean(image[:,:,100:150,100:150],(1,2,3))
		image = image[mean_intensities<0.89]
		norm_img = self.normalizer(image)
		return norm_img
	
	def _get_target_tensor(self):
		target_tensor=load('stain_matrices/target_tcga_hgg.pt')
		return target_tensor

	#dont have to make it a module
	def to(self,dev):
		self.normalizer = self.normalizer.to(dev)

	def randomize(self):
		#the torch-stain augmenters have built in augmentation every batch
		pass
