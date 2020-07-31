import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F
import torch.nn.init as init

class MLP(nn.Module):
	"""
	Two layer MLP for MNIST benchmarks.
	"""
	def __init__(self, hiddens, output, config):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(784, hiddens)    
		self.relu = nn.ReLU(inplace=True)
		self.W2 = nn.Linear(hiddens, hiddens)
		self.W3 = nn.Linear(hiddens, 10)
		self.dropout_p = config['dropout']

	def forward(self, x, task_id=None):
		# x = x.view(-1, 784 + self.num_condition_neurons)
		out = self.W1(x)
		out = self.relu(out)
		out = nn.functional.dropout(out, p=self.dropout_p)
		out = self.W2(out)
		out = self.relu(out)
		out = nn.functional.dropout(out, p=self.dropout_p)
		out = self.W3(out)
		return out




class GatedMLP(nn.Module):
	def __init__(self, hiddens=100):
		super(GatedMLP, self).__init__()
		self.gates = nn.ModuleDict()
		for i in range(5):
			gate_skeleton = nn.Sequential(OrderedDict([
				('layer1', weightNorm(nn.Linear(784, hiddens))),
				('act1', nn.ReLU()),
				('layer2', weightNorm(nn.Linear(hiddens, hiddens))),
				('act2', nn.ReLU()),
				('layer3', weightNorm(nn.Linear(hiddens, 10))),
			]))
			self.gates['gate{}'.format(i+1)] =  gate_skeleton
		# self.gates = nn.Module(OrderedDict(gates))

	def forward(self, x, task_id):
		gate = 'gate{}'.format(task_id)
		out = self.gates[gate](x)
		return out


		
def conv3x3(in_planes, out_planes, stride=1):
	" 3x3 convolution with padding "
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion=1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = weightNorm(conv3x3(inplanes, planes, stride))
		# self.bn1 = nn.BatchNorm2d(planes)
		self.gn1 =nn.GroupNorm(inplanes, planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = weightNorm(conv3x3(planes, planes))
		# self.bn2 = nn.BatchNorm2d(planes)
		self.gn2 =nn.GroupNorm(inplanes, planes)

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.gn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.gn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out



class Resnet(nn.Module):

	def __init__(self, block, layers, num_classes=10):
		super(Resnet, self).__init__()
		self.inplanes = 16
		self.conv1 = weightNorm(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2, bias=False))
		# self.bn1 = nn.BatchNorm2d(16)
		self.gn1 = nn.GroupNorm(1, 16)
		self.relu = nn.ReLU(inplace=True)
		self.layer1 = self._make_layer(block, 16, layers[0])
		self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.avgpool = nn.AvgPool2d(8, stride=1)
		self.fc = weightNorm(nn.Linear(64 * block.expansion, num_classes))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion)
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x, task_id=None):
		x = self.conv1(x)
		x = self.gn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def Resnet20(**kwargs):
	model = Resnet(BasicBlock, [3, 3, 3], **kwargs)
	return model
