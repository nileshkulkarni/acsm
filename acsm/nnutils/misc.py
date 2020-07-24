import torch


def add_bIndex(faces):
    bsize = len(faces)
    bIndex = torch.LongTensor([i for i in range(bsize)]).type(faces.type())
    bIndex = bIndex.view(-1, 1, 1).repeat(1, faces.shape[1], 1)
    faces = torch.cat([bIndex, faces], dim=-1)
    return faces


def get_img_grid(img_size):
    x = torch.linspace(-1, 1, img_size[1]).view(1, -1).repeat(img_size[0], 1)
    y = torch.linspace(-1, 1, img_size[0]).view(-1, 1).repeat(1, img_size[1])
    grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
    grid.unsqueeze(0)
    return grid