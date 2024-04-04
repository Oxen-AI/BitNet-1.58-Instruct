
from bitnet.models.bitnet.causal_lm import BitLinear
import torch

def main():
    
    layer = BitLinear(3, 5, bias=False)
    layer.weight = torch.nn.Parameter(torch.tensor([
        [2.2, 2.1, 2.4],
        [-2.2, -2.1, -2.7],
        [2.8, 2.3, 2.2],
        [0., 0., 0.],
        [2.5, 2.3, 2.6]
    ]))
    x = torch.tensor([3., 2., 1.])
    y = layer(x)
    print(y)

if __name__ == '__main__':
    main()