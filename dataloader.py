import torch
from torch.utils.data import DataLoader
import cv2
import os
from torchvision import transforms
from PIL import Image
def image_list(path="~/Downaloads/", type="real_images/"):
    data_list = []
    for image in sorted(os.listdir(path + type)):
        data_list.append(os.path.join(path + type, image ))

    return data_list




transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class image_dataset(DataLoader):

    def __init__(self, path, transforms=transform, type = "real_images/") :

        self.real_image_list = image_list(path=path, type=type)
        self.transforms = transforms
        print("# of training real images samples:", len(self.real_image_list))

    def __getitem__(self, index):

        img_path_list = self.real_image_list[index]

        img = cv2.imread(img_path_list)[:,:,(2,1,0)]


        return  self.transforms(img)

    def __len__(self):
        return len(self.real_image_list)



# data = image_dataset(path="/home/yachao-li/Downloads/")
# dataloader = torch.utils.data.DataLoader(data,128)
# for i in dataloader:
#     print(i.size())



