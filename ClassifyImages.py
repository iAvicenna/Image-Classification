################################################################################
#This is a classification code (works on GPU) to classify images from CIFAR10. #
#This database contains images of size 3 x 32 x 32 (50000 training and 10000   #
#test examples) with 10 classes ('plane', 'car', 'bird', 'cat', 'deer', 'dog', #
# 'frog', 'horse', 'ship', 'truck'). The architecture is:                      #
#                                                                              #
# Input -> (Relu(Conv) -> MaxPool)*4 -> Relu(Full)*3 -> Output                 #
#                                                                              #
# Over several runs the best achieved accuracy over the test set is %70        #
# percent.                                                                     #                                                
################################################################################

#Reset the workspace and import variables

%reset -f 

import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


#these are the per channel mean and standart deviation of
#CIFAR10 image database. We will use these to normalize each
#channel to unit deviation with mean 0.

mean_CIFAR10=np.array([0.49139968, 0.48215841, 0.44653091])
std_CIFAR10=np.array([0.49139968, 0.48215841, 0.44653091])

#function below is used to visualize
#CIFAR10 object represented as a [3][32][32] torch
#tensor. It first changes the type to numpy then
#reverts the normalization applied to the dataset 
#and then transposes the first and last dimension
def imshow(img):
    a=img.numpy();
    b = np.reshape(mean_CIFAR10, (3,1,1))
    c = np.reshape(std_CIFAR10, (3,1,1))
    
    
    d=np.add(np.multiply(c,a),b) #multplication and addition is done 
                                 #via broadcasting
    
    d=np.transpose(d,(2, 1, 0)) #change shape
    
    
    imgplot = plt.imshow(d)
 
#this transformation is used to transform the images to 0 mean and 1 std.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean_CIFAR10 , std_CIFAR10)])

#load the CIFAR10 training and test sets
training_set_CIFAR10 = datasets.CIFAR10(root = 'cifar10/',
                                  transform = transform,
                                  train = True,
                                  download = True)


test_set_CIFAR10 = datasets.CIFAR10(root = 'cifar10/',
                                  transform = transform,
                                  train = False,
                                  download = True)

print('Number of training examples:', len(training_set_CIFAR10))
print('Number of test examples:', len(test_set_CIFAR10))

#there are ten classes in the CIFAR10 database
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#DataLoaders are used to iterate over the database images in batches rather
#one by one using for loops which is expensive in python since it is interpreted
training_loader_CIFAR10 = torch.utils.data.DataLoader(dataset=training_set_CIFAR10,
                                              batch_size=10,
                                              shuffle=True)

test_loader_CIFAR10 = torch.utils.data.DataLoader(dataset=test_set_CIFAR10,
                                            batch_size=10,
                                            shuffle=False)

#this function is used to test the accuracy of the model     
#over the test set. The network cnn is defined later on in the code.
def test():
    print('Started evaluating test accuracy...')
    cnn.eval()
    #calculate the accuracy of our model over the whole test set in batches
    correct = 0
    for x, y in test_loader_CIFAR10:
        x, y = Variable(x).cuda(), y.cuda()
        h = cnn.forward(x)
        pred = h.data.max(1)[1]
        correct += pred.eq(y).sum()
    return correct/len(test_set_CIFAR10)



#Below we define the convolutional network class.
#See the beginning of the document for the architecture

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        #define the feature extraction layers
        self.conv1 = torch.nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)   
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4 = torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        #define the deep learning layers
        self.full1=nn.Linear(512,512)        
        self.full2=nn.Linear(512, 256)        
        self.full3=nn.Linear(256,10)
        

       
    #define the forward run for the input data x    
    def forward(self, x):
    
        #convolutional feature extraction layers
        x = F.relu(self.conv1(x))   
        x = self.pool1(x)
        x = F.relu(self.conv2(x))   
        x = self.pool2(x)
        x = F.relu(self.conv3(x))   
        x = self.pool3(x)
        x = F.relu(self.conv4(x))   
        x = self.pool4(x)
        
        
        #learning layers
        x = x.view(-1,512)
        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        x = F.softmax(self.full3(x),dim=1)
        
        return x
        

#this is the training function. cnn is the network that is defined later
#optimizer and learning rate lr are modified inside the function

def train(cycles,cost_criterion,cnn,optimizer):
    
    average_cost=0 #cost function for the training
    acc=0 #accuracy over the test set

    
    for e in range(cycles): #cycle through the database many times

        print('Cycle: ',e)
        cnn.train()
         
        #following for loop cycles over the training set in batches
        #of batch_number=5 using the training_loader object
        for i, (x, y) in enumerate(training_loader_CIFAR10 ,0):
        
            #here x,y will store data from the training set in batches 
            x, y = Variable(x).cuda(), Variable(y).cuda()
            

            h = cnn.forward(x) #calculate hypothesis over the batch
            
            cost = cost_criterion(h, y) #calculate cost the cost of the results
            
            optimizer.zero_grad() #set the gradients to 0
            cost.backward() # calculate derivatives wrt parameters
            optimizer.step() #update parameters

            average_cost=average_cost+cost.data[0]; #add the cost to the costs
            
            
            if i % 200 == 199: #print cost on each 200 iterations
              print('\tBatch', i, '\tCost', average_cost/200)
              average_cost=0;
        
        acc = test() #once a training over the whole dataset is complete 
                     #look at the accuracy before reiterating the training
        print('\tTest accuracy: ', acc)    


cycles = 50 #number of cycles that the training runs over the database
cost_criterion = torch.nn.CrossEntropyLoss() #cost function
cnn = ConvNet().cuda() #build the initial network (in the GPU)
optimizer=optim.Adam(cnn.parameters(), lr= 0.0001)

train(cycles,cost_criterion,cnn,optimizer)
torch.save(cnn.state_dict(), 'cnn_trained')
                    
