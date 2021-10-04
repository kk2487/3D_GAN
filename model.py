import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

def conv3x3x3(in_planes, out_planes, stride=1):
	# 3x3x3 convolution with padding
	return nn.Conv3d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=1,
		bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3x3(planes, planes)
		self.bn2 = nn.BatchNorm3d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class Discriminator(nn.Module):

	def __init__(self,block,layers,opt):
		self.shortcut_type = 'B'
		self.sample_size = 224
		self.sample_duration = 10
		self.num_classes = 1
		self.inplanes = 64
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv3d(
			1,
			64,
			kernel_size=(1,7,7),
			stride=(1, 2, 2),
			padding=(3, 3, 3),
			bias=False)
		self.bn1 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0], self.shortcut_type)
		self.layer2 = self._make_layer(
			block, 128, layers[1], self.shortcut_type, stride=2)
		self.layer3 = self._make_layer(
			block, 256, layers[2], self.shortcut_type, stride=2)
		self.layer4 = self._make_layer(
			block, 512, layers[3], self.shortcut_type, stride=2)
		last_duration = int(math.ceil(self.sample_duration / 16))
		last_size = int(math.ceil(self.sample_size / 32))
		self.avgpool = nn.AvgPool3d(
			(last_duration, last_size, last_size), stride=1)
		self.fc = nn.Linear(512 * block.expansion, self.num_classes)
		self.sig = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			if shortcut_type == 'A':
				downsample = partial(
					downsample_basic_block,
					planes=planes * block.expansion,
					stride=stride)
			else:
				downsample = nn.Sequential(
					nn.Conv3d(
						self.inplanes,
						planes * block.expansion,
						kernel_size=1,
						stride=stride,
						bias=False), nn.BatchNorm3d(planes * block.expansion))

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = x.unsqueeze(1)
		#print("-----------------")
		#print(x)
		x = self.conv1(x)

		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		lookshape = False
		x = self.layer1(x)
		if lookshape:
			print('\nlayer1-------------')
			print(np.shape(x))
			print('--------------')
		x = self.layer2(x)
		if lookshape:
			print('\nlayer2-------------')
			print(np.shape(x))
			print('--------------')
		x = self.layer3(x)
		if lookshape:
			print('\nlayer3-------------')
			print(np.shape(x))
			print('--------------')
		x = self.layer4(x)
		if lookshape:
			print('\nlayer4-------------')
			print(np.shape(x))
			print('--------------')

		x = self.avgpool(x)

		x = x.view(x.size(0), -1)
		x = self.fc(x)
		#print("---------------")
		#print(x)
		x = self.sig(x)
		#print(x)
		return x
"""
class Discriminator(nn.Module):
	def __init__(self, opt):
		super(Discriminator, self).__init__()
		
		self.model = ResNet(BasicBlock, [3, 4, 6, 3])
		# return model
"""
"""
class Discriminator(nn.Module):
	def __init__(self, opt):
		super(Discriminator, self).__init__()
		#Input shape= (10,224,224)		
		self.conv1=nn.Conv2d(in_channels=10,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,224,224)
		self.relu1=nn.ReLU()
		#Shape= (32,224,224)
		self.pool1=nn.MaxPool2d(kernel_size=2)
		#Shape= (32,112,112)
		
		
		self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,112,112)
		self.relu2=nn.ReLU()
		#Shape= (64,112,112)
		self.pool2=nn.MaxPool2d(kernel_size=2)
		#Shape= (64,56,56)
		
		
		self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,56,56)
		self.relu3=nn.ReLU()
		#Shape= (128,56,56)
		self.pool3=nn.MaxPool2d(kernel_size=2)
		#Shape= (128,28,28)        
		
		self.conv4=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,28,28)
		self.relu4=nn.ReLU()
		#Shape= (64,28,28)
		self.pool4=nn.MaxPool2d(kernel_size=2)
		#Shape= (64,14,14)   
		
		self.conv5=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,14,14)
		self.relu5=nn.ReLU()
		#Shape= (32,14,14)
		self.pool5=nn.MaxPool2d(kernel_size=2)
		#Shape= (32,7,7) 
		
		self.fc1=nn.Linear(in_features=7 * 7 * 32,out_features=1024)
		self.relu6=nn.ReLU()

		self.fc2=nn.Linear(in_features=1024,out_features=512)
		self.relu7=nn.ReLU()

		self.fc3=nn.Linear(in_features=512,out_features=1)

		self.sig=nn.Sigmoid()

	def forward(self, input):
		output=self.conv1(input)
		output=self.relu1(output)
		output=self.pool1(output)
		  
			
		output=self.conv2(output)
		output=self.relu2(output)
		output=self.pool2(output)
		
		output=self.conv3(output)
		output=self.relu3(output)
		output=self.pool3(output)
			
		output=self.conv4(output)
		output=self.relu4(output)
		output=self.pool4(output)    
		
		output=self.conv5(output)
		output=self.relu5(output)
		output=self.pool5(output)
		
		output=output.view(-1, 7 * 7 * 32)        
		
		output=self.fc1(output)
		output=self.relu6(output)
		
		output=self.fc2(output)
		output=self.relu7(output)
		
		output=self.fc3(output)
		output=self.sig(output)
			
		return output

"""
class Generator(nn.Module):
	def __init__(self, opt):
		super(Generator, self).__init__()
		
		# encoder ---------------------------------------------------------------------------------

		#Input shape= (1,224,224)		
		self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,224,224)
		self.relu1=nn.ReLU()
		#Shape= (32,224,224)
		self.pool1=nn.MaxPool2d(kernel_size=2, return_indices=True)
		#Shape= (32,112,112)
		
		
		self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,112,112)
		self.relu2=nn.ReLU()
		#Shape= (64,112,112)
		self.pool2=nn.MaxPool2d(kernel_size=2, return_indices=True)
		#Shape= (64,56,56)
		
		
		self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,56,56)
		self.relu3=nn.ReLU()
		#Shape= (128,56,56)

		self.conv4=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,56,56)
		self.relu4=nn.ReLU()
		#Shape= (128,56,56)
	   
		self.conv5=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,56,56)
		self.relu5=nn.ReLU()
		#Shape= (128,56,56)
		
		# decoder ---------------------------------------------------------------------------------
		self.convt1= nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,56,56)
		self.relu6=nn.ReLU()

		self.convt2= nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (64,56,56)
		self.relu7=nn.ReLU()

		
		#Shape= (128,56,56)
		self.convt3= nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,56,56)
		self.relu8=nn.ReLU()

		self.unpool1=nn.MaxUnpool2d(kernel_size=2)
		#Shape= (64,112,112)
		self.convt4=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,112,112)
		self.relu9=nn.ReLU()

		self.unpool2=nn.MaxUnpool2d(kernel_size=2)
		#Shape= (64,224,224)
		self.convt5=nn.ConvTranspose2d(in_channels=32,out_channels=10,kernel_size=3,stride=1,padding=1)
		#Shape= (10,224,224)
		self.relu10=nn.ReLU()
		self.tan=nn.Tanh()

	def forward(self, input):
		
		# encoder ---------------------------------------------------------------------------------
		#print("---------- encoder ----------")
		#print(input.shape)
		output=self.conv1(input)
		output=self.relu1(output)
		output, indices1 =self.pool1(output)
		#print(output.shape)

		output=self.conv2(output)
		output=self.relu2(output)
		output, indices2=self.pool2(output)
		#print(output.shape)

		output=self.conv3(output)
		output=self.relu3(output)
		#print(output.shape)

		output=self.conv4(output)
		output=self.relu4(output)

		output=self.conv5(output)
		output=self.relu5(output)

		# decoder ---------------------------------------------------------------------------------
		#print("---------- decoder ----------")
		
		output=self.convt1(output)
		output=self.relu6(output)

		output=self.convt2(output)
		output=self.relu7(output)

		output=self.convt3(output)
		output=self.relu8(output)
		#print(output.shape)

		output=self.unpool1(output,indices2)
		output=self.convt4(output)
		output=self.relu9(output)
		#print(output.shape)

		output=self.unpool2(output,indices1)
		output=self.convt5(output)
		output=self.tan(output)
		#print(output.shape)

		return output

if __name__ == '__main__':
	import argparse
	def parse_opts():
		parser = argparse.ArgumentParser()
		parser.add_argument("--mode", type=str, default="test", help="train, test")
		parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
		parser.add_argument("--save_interval", type=int, default=20, help="number of epochs of training")
		parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
		parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
		parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
		parser.add_argument("--b2", type=float, default=0.7, help="adam: decay of first order momentum of gradient")
		parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
		parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
		parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
		parser.add_argument("--channels", type=int, default=1, help="number of image channels")
		parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
		parser.add_argument("--dataset", type=str, default="C:/Users/hongze/3d_resnet/train_data/test/", help="data path")
		args = parser.parse_args()
		return args

	opt = parse_opts()
	from torchsummary import summary
	model = Discriminator(BasicBlock, [3, 4, 6, 3], opt = opt).cuda()
	summary(model, (10, 224, 224))