from torch import nn
import torch
from torch.nn import functional as F


class ResNet(nn.Module):
	def __init__(self, in_flow=6, out_flow=2, ext_dim=19):
		super(ResNet, self).__init__()
		self.close = resunit(in_flow, out_flow)
		self.period = resunit(in_flow, out_flow)
		self.trend = resunit(in_flow, out_flow)
		self.ext = nn.Sequential(
			nn.Linear(in_features=ext_dim, out_features=10),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=10, out_features=2 * 32 * 32))

		self.w1 = nn.Parameter(-torch.ones((2, 32, 32), requires_grad=True).double())
		self.w2 = nn.Parameter(-torch.ones((2, 32, 32), requires_grad=True).double())
		self.w3 = nn.Parameter(-torch.ones((2, 32, 32), requires_grad=True).double())

	def forward(self, close_data, period_data, trend_data, feature_data):
		m = nn.ReLU()
		close_out = self.close(close_data)
		period_out = self.period(period_data)
		trend_out = self.trend(trend_data)
		ext_out = self.ext(feature_data)
		ext_out = ext_out.view(2, 32, 32)
		main_out = torch.mul(close_out, self.w1) + torch.mul(period_out, self.w2) + torch.mul(trend_out,
																							  self.w3) + ext_out
		return main_out


class resunit(nn.Module):
	def __init__(self, in_flow=6, out_flow=2):
		super(resunit, self).__init__()
		self.unit = nn.Sequential(
			nn.Conv2d(in_flow, 64, kernel_size=3, stride=1, padding=1, bias=False),
			residual(),
			residual(),
			residual(),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, out_flow, kernel_size=3, stride=1, padding=1, bias=False))

	def forward(self, x):
		return self.unit(x)


class residual(nn.Module):
	def __init__(self, in_flow=64, out_flow=64):
		super(residual, self).__init__()
		self.left = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(in_flow, out_flow, kernel_size=3, stride=1, padding=1, bias=False)
		)

	def forward(self, x):
		out = self.left(x)
		out = self.left(out)
		res = x
		out = out + x
		return out


if __name__ == '__main__':
	a = ResNet()
	para = list(a.parameters())
	for i in range(0, len(para)):
		print(para[i].size())
	print(a.w1)
