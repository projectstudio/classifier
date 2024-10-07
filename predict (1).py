import time
import json
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from workspaceutils import active_session
import argparse
from collections import OrderedDict

#define possible network structure struct
network_struct = {"densenet121" : 1024,
                 "alexnet" : 9216,
                 "vgg16":25088}


def parse_command_line_arguments():
    """
    Defines and parses command-line arguments (both positional and optional)

    :return: parsed object to be used to extract arguments values
    """
    parser = argparse.ArgumentParser()

    # Positional args
    parser.add_argument('data_directory', action="store")

    # Optional args
    parser.add_argument('--save_dir', action='store',
                        dest='save_dir',
                        help='Load categories names from given file',
                        default="checkpoint.pth")

    parser.add_argument('--gpu', action='store_true',
                        dest='device',
                        help='Device of prediction processing',
                        default=False)

    parser.add_argument('--arch', action='store',
                        dest='arch',
                        help='Name of pre-trained network used for training',
                        default="densenet121")

    parser.add_argument('--learning_rate', action='store',
                        dest='learning_rate',
                        help='value of training learning rate',
                        default=0.001)

    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units',
                        help='Number of units in the fully-connected hidden '
                             'layer of the neural netwrork',
                        default=512)

    parser.add_argument('--epochs', action='store',
                        dest='epochs',
                        help='Number of training epochs',
                        default=5)
    
    parser.add_argument('--catalog', action='store',
                        dest='catalog',
                        help='Catalog to name',
                        default="cat_to_name")
    
    parser.add_argument('--dropout', action='store',
                        dest='dropout',
                        help='value of dropout',
                        default=0.5)


    # Parse all args
    results = parser.parse_args()
    
    return results


def load_data(data_dir):
    """
    Loads data for training and validation

    :param data_directory: str; directory of images

    :return: data loaders objects for training and validation sets
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # define my transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(test_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=16, shuffle=True)


    return {"trainloader": trainloader,
            "validationloader": validationloader,
            "testloader": testloader ,
            "class_to_idx": train_data.class_to_idx}


def load_catalog(cat_filename):

    with open(cat_filename, 'r') as f:
        cat_to_name = json.load(f)
    
    return (cat_to_name)


#%%
# TODO: Build and train your network
from collections import OrderedDict
network_struct = {"densenet121" : 1024,
                 "alexnet" : 9216,
                 "vgg16":25088}

def network_setup(nstruct='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001):
    print("Getting the model")
    if(nstruct=="densenet121"):
        model = models.densenet121(pretrained=True)
    elif(nstruct=="alexnet"):
        model = models.alexnet(pretrained=True)
    elif(nstruct=="vgg16"):
        model = models.vgg16(pretrained=True)
    else :
        print("The network stucure type {} yoy chose is unavailale type,  allowed are :densenet121, alexet, vgg16.".format(network_struct))
    
    print("freezing the layers")    
    for param in model.parameters():
        param.requires_grad = False

    print("Defining the classifier")
    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(dropout)),
        ('inputs', nn.Linear(network_struct[nstruct], hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_layer1, 256)),
        ('relu2',nn.ReLU()),
        ('hidden_layer3',nn.Linear(256,102)),
        ('output', nn.LogSoftmax(dim=1))
                      ]))
        
    print("Optimers")    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    print("Returing the model")    
    return model , optimizer ,criterion 

    
# Defining validation 
def validation(model, valid_loader, criterion,device):
    model.to(device)
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

#define train network
def train_net (device,epochs,model,criterion,trainloader):

    print_every = 50
    steps = 0
    loss_show=[]
    model.to(device)
    print("Start Training")

    start_train = time.time()

    for e in range(epochs):
        model.train()
        running_loss = 0
        start = time.time()
        for il, (inputs, labels) in enumerate(trainloader):
            steps += 1
            print (steps)
            print (round((time.time()-start)/60,2)," minutes")
            
            inputs,labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval () 
                with torch.no_grad():
                    validationlost, accuracy = validation(model, validationloader, criterion,device)
                        
                validationlost = validationlost / len(validationloader)
                accuracy = accuracy /len(validationloader)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(validationlost),
                      "Accuracy: {:.4f}".format(accuracy))
                
                
                running_loss = 0
                model.train()
        end = time.time()
        print("Time taken for Epoch: ", e," is ",round((end-start)/60,2)," minutes")
        
        print("Model Trained")
        print ("Total time for train:" ,round((time.time()-start_train)/60,2)," minutes")
    
        return model


#  Accuracy test

def check_accuracy_on_test(testloader, model):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def load_model(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    nstruct = checkpoint['nstruct']
    hidden_layer1 = checkpoint['hidden_layer1']
    droput = checkpoint['dropout']
    lr = checkpoint['learning_rate']

    model, _, _ = network_setup(nstruct, dropout, hidden_layer1)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    img_pil = Image.open(image)

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = adjustments(img_pil)

    return img_tensor

# TODO: Display an image along with the top 5 classes
import seaborn as sns
def check_sanity(image):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image.split('/')[2]
    title_ = cat_to_name[flower_num]
    print(title_)
    # Plot flower
    img = process_image(image)
    imshow(img, ax, title = title_);
    # Make prediction
    labs, flowers,probs= predict(image, model)
    #print(labs,probs)
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0],orient = 'h');
    plt.show()


if __name__ == "__main__":
    """
    Command-line usage example:

    $ python train.py data_directory

    Options: (all or nothing; any order)
        --save_dir save_directory
        --arch "vgg13"
        --learning_rate 0.01
        --dropout 0.5
        --hidden_units 512
        --epochs 20
        --gpu
        --catalog

    Prints out training loss, validation loss, and validation
    accuracy as the network trains 
    AND when done training, it saves the trained model checkpoint
    """
    cmd_arguments = parse_command_line_arguments()
    
    data_directory = cmd_arguments.data_directory
    save_dir = cmd_arguments.save_dir
    device = "cuda" if cmd_arguments.device else "cpu"
    nstruct = cmd_arguments.arch
    lr = float(cmd_arguments.learning_rate)
    dropout =  float(cmd_arguments.dropout)                   
    hidden_layer1 = int(cmd_arguments.hidden_units)
    epochs = int(cmd_arguments.epochs)
    catalog = cmd_arguments.catalog

    # load data
    print('* Loading data in progress ...')
    dataloaders = load_data(data_directory)
    trainloader = dataloaders["trainloader"]
    testloader =  dataloaders["testloader"]
    validationloader = dataloaders["validationloader"]
    # testloader = dataloaders["testloader"]
    class_to_idx = dataloaders["class_to_idx"]
    print('* Data loaded successfully!\n')


    cat_to_name = load_catalog(catalog+'.json')
   #Build network
    model,optimizer,criterion = network_setup (nstruct,dropout, hidden_layer1,lr)

    loaded_model = load_model('checkpoint.pth')
    # print(loaded_model)
    #check_accuracy_on_test(testloader, loaded_model)


    image = (data_directory + '/test' + '/1/' + 'image_06752.jpg')
    print(image)
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    
    img = process_image(image)
    print(img.shape)

    print(image)
    model.class_to_idx = train_data.class_to_idx
    ctx = model.class_to_idx
    check_sanity(image)