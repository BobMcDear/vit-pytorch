from torch import randn

from transformer import VisionTransformer


if __name__ == '__main__':
    xb = randn(8, 3, 256, 256)
    vit = VisionTransformer(512,16,256,6,64,8,0.2,2000,1)
    s = 0
    for p in vit.parameters():
        s += p.numel()
    print(s)
