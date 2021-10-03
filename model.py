import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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
		self.pool3=nn.MaxPool2d(kernel_size=2, return_indices=True)
		#Shape= (128,28,28)        
		
		# decoder ---------------------------------------------------------------------------------
		
		self.unpool1=nn.MaxUnpool2d(kernel_size=2)
		#Shape= (128,56,56)
		self.convt1= nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,56,56)
		self.relu4=nn.ReLU()

		self.unpool2=nn.MaxUnpool2d(kernel_size=2)
		#Shape= (64,112,112)
		self.convt2=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,112,112)
		self.relu5=nn.ReLU()

		self.unpool3=nn.MaxUnpool2d(kernel_size=2)
		#Shape= (64,224,224)
		self.convt3=nn.ConvTranspose2d(in_channels=32,out_channels=10,kernel_size=3,stride=1,padding=1)
		#Shape= (10,224,224)
		self.relu6=nn.ReLU()
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
		output, indices3=self.pool3(output)
		#print(output.shape)

		# decoder ---------------------------------------------------------------------------------
		#print("---------- decoder ----------")
		output=self.unpool1(output,indices3)
		output=self.convt1(output)
		output=self.relu4(output)
		#print(output.shape)

		output=self.unpool2(output,indices2)
		output=self.convt2(output)
		output=self.relu5(output)
		#print(output.shape)

		output=self.unpool3(output,indices1)
		output=self.convt3(output)
		output=self.tan(output)
		#print(output.shape)

		return output