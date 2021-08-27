

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import copy
import torch
import torchvision.models as models



import os
os.chdir("/Users/Derek/Desktop/NST")


def image_loader(image_path, im_size):
    loader = transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


#sets normalization bounds according to best practices in literature

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std



#super class that allow us to use this as part of the model later
#the content pic is detached as this is a static value and we want to train the other portion

class content_loss(nn.Module):
    def __init__(self, content_pic):
        super(content_loss, self).__init__()
        
        self.content_pic = content_pic.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.content_pic)
        return input


#define the gram matrix to be used later in the style loss function

def gram_matrix(input):
    a, b, c, d = input.size()
    new = torch.reshape(input, (a*b, c*d))
    final = torch.mm(new, new.t())
    return final.div(a * b * c * d)




#creates a sub class of the nn.module in order to be able to put in the model - the forward method just returns itself
# so does not hurt calculations
class style_loss_module(nn.Module):
    def __init__(self, style_pic):
        super(style_loss_module, self).__init__()
        self.style_pic_gram = gram_matrix(style_pic).detach()
    
    def forward(self, input):
        inp_gram = gram_matrix(input)
        self.loss = F.mse_loss(inp_gram, self.style_pic_gram)
        
        return input
    


#creates a copy of the cnn attaching content and style loss & replacing max pool

def create_model(transfer_model, content_pic, style_pic): 
    
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to('cpu')
    model = nn.Sequential(normalization)
    
    conv_counter = 0
    counter = 0
    style_loss_list = []
    content_loss_list = []

    
    for i in transfer_model.children():
        counter +=1 
        if isinstance(i, nn.MaxPool2d):
            model.add_module("avg_pool" + str(counter), nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
            
        if isinstance(i, nn.ReLU):
            model.add_module("Relu" + str(counter), nn.ReLU(inplace=False))
            
        if  isinstance(i, nn.Conv2d):
            conv_counter += 1
            model.add_module("Conv" + str(counter), i)
            
            if conv_counter == 4:
                content_inp = model(content_pic)
                c_loss = content_loss(content_inp)
                model.add_module("content_loss" + str(i), c_loss)
                content_loss_list.append(c_loss)
                
            if conv_counter >= 1 and conv_counter < 6:
                style_inp = model(style_pic)
                style_loss = style_loss_module(style_inp)
                model.add_module("style_loss" + str(i), style_loss)
                style_loss_list.append(style_loss)
        if conv_counter > 6:
            break
                
            
    return style_loss_list, content_loss_list, model
 


#returns the optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr = .1)
    return optimizer



#main function that takes all inputs and outputs a new image
#adjust the style and content weight in order to get an image according to your liking
def nst(base_im, content_im, style_im, transfer_model, steps = 90, style_weight = 1000000, content_weight = 1):
    
    base_size = Image.open(content_im).convert('RGB')
    
    
    image = image_loader(content_im, 256)
    image2 = image_loader(style_im, 256)
    
    style_losses, content_losses, model = create_model(transfer_model, image, image2)
    
    #sets the image to the content image if not specified
    if base_im == 'null':
        base_im = image
    
    optimizer = get_input_optimizer(base_im)
    run = [0]
    
    #runs for the specified amount of steps
    for epoch in range(steps):
        if run[0] > steps:
            break
        def closure():
            # correct the values of updated input image
            base_im.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(base_im)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    base_im.data.clamp_(0, 1)
    return base_im

if __name__ == "__main__":
    device = "cpu"
    
    # load the vgg model and set the device to run calculations
    transfer_model = models.vgg19(pretrained=True).features.eval()
    image1_dir = ''
    image2_dir = ''
    
    new = nst('null', image1_dir, image2_dir, transfer_model, steps = 5)
    fun = transforms.ToPILImage()

    image = new.cpu().clone()
    image = image.squeeze(0)
    image = fun(image)
    plt.imshow(image)
    
                




