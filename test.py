import numpy as np
import torch




h = torch.rand((1,62,200)).transpose(1, 2)

a = torch.rand((62,62))

b = torch.matmul(h, a).transpose(1, 2)




pass