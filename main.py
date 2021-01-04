# # %%
# #from model_arl import LocalConv2dReLU, HierarchicalMultiScaleRegionLayer,HMRegionLearning,ChannelWiseSpatialAttentLearning
# import torch.optim as optim
# from PIL import Image
# from torchvision import transforms
# from fcn8s import Fcn8s
# from crfasrnn_model import CrfRnnNet

# img2 = Image.open(r'C:\Users\Yaqian\Downloads\JinHyunCheong.jpg')
# img2 = transforms.ToTensor()(img2)
# img2 = img2[None,:,:,:]
# cropped_image = img2[:,:, 0:176, 0:176]
# local_grp = [[2,2],[4,4],[8,8]]
# #net0 = ChannelWiseSpatialAttentLearning(3,24,1,1,0)
# #net0 = LocalConv2dReLU(8,8,3,32,3)
# net0 = CrfRnnNet(n_class=12)
# print(net0)

# #outputs = net0(cropped_image)

# %%
import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from fcn8s import Fcn8s
from crfasrnn_model import CrfRnnNet
from crfrnn import CrfRnn
net0 = CrfRnn(num_labels = 2, num_iterations = 5)
marka = torch.rand((16,176,176))
markb = torch.rand((1,176,176))
outputs = net0(marka,markb)
#v_probab = torch.div(outputs[0,1,:,:], (outputs[0,1,:,:]+outputs[0,0,:,:]))

# %%
def calculate_AU_weight(occurence_df):
    """
    Calculates the AU weight according to a occurence dataframe 
    inputs: 
        occurence_df: a pandas dataframe containing occurence of each AU. See BP4D+
    """
    #occurence_df = occurence_df.rename(columns = {'two':'new_name'})
    #occurence_df2 = occurence_df.iloc[:,2::]
    occurence_df2 = occurence_df[['1','2', '4','6','7','10','12','14','15','17','23','24']]
    weight_mtrx = np.zeros((occurence_df2.shape[1], 1))
    for i in range(occurence_df2.shape[1]):
        weight_mtrx[i] = np.sum(occurence_df2.iloc[:, i]
                                > 0) / float(occurence_df2.shape[0])
    weight_mtrx = 1.0/weight_mtrx

    #print(weight_mtrx)
    weight_mtrx[weight_mtrx == np.inf] = 0
    #print(np.sum(weight_mtrx)*len(weight_mtrx))
    weight_mtrx = weight_mtrx / (np.sum(weight_mtrx)*len(weight_mtrx))

    return(weight_mtrx)

def dice_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)
    
def au_dice_loss(input, target, weight=None, smooth = 1, size_average=True):
    for i in range(input.size(2)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, 1, i]).exp()
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        t_loss = dice_loss(t_input, t_target, smooth)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()

def train_val_dataset(dataset,val_split = 0.25):
    train_idx,val_idx = train_test_split(list(range(len(dataset))),test_size = 0.25)
    datasets = {}
    datasets['train'] = Subset(dataset,train_idx)
    datasets['test'] = Subset(dataset,val_idx)
    return(datasets)


Dataset01 = image_Loader(csv_dir="F:\\here.csv", img_dir="F:\\FaceExprDecode\\F001\\", transform=None, action_unit='6')
train_set,test_set = torch.utils.data.random_split(Dataset01,[1000,230])
train_loader = DataLoader(dataset=train_set,batch_size=50,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=50,shuffle=True)

import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, momentum=0.9)
