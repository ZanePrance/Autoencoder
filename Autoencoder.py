import datetime
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
import torchsummary
from torchsummary import summary
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print("training...")
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        # iterate through each of the batches of images in the train_loader
        for batch in train_loader:
            img, label = batch  # go through each img and label in the current batch
            img = img.to(device=device) # use the cpu
            img = img.view(img.size(0), -1)
            outputs = model(img)    # propagate the current image through the neural network
            # flatten the img to a 1D tensor of size (batch_size, 784), which is the same as the input to the network

            loss = loss_fn(outputs, img)    # calculate the training loss
            optimizer.zero_grad()   # reset the optimizer gradients to 0
            loss.backward()     # calculate the loss gradients
            optimizer.step()    # iterate the optimization, based on the loss gradients
            loss_train += loss.item()   # update the value of the losses
        scheduler.step(loss_train)  # have the optimizer update some of the hyperparameters

        losses_train += [loss_train / len(train_loader)]    # update the loss value
        # print the time each epoch finished training as well as the current training loss
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
        # return the training loss, so it can be plotted and displayed to the user after the call
    return losses_train

class autoencoderMLP4Layer(nn.Module):
    def __init__(self, N_input=784, N_bottleneck = 8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)   # input = 1*748, output = 1*392
        self.fc2 = nn.Linear(N2, N_bottleneck)  # output = 1*N
        self.fc3 = nn.Linear(N_bottleneck, N2)  # output = 1*392
        self.fc4 = nn.Linear(N2, N_output)  # output = 1*784
        self.type = 'MLP4'
        self.input_shape = (1,28*28)

    def encode(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def decode(self, x):
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


def test_reconstruction(model, test_loader, index):
    model.load_state_dict(torch.load("MLP.8.pth"))
    model.eval()

    with torch.no_grad():
        for imgs, label in test_loader:
            imgs =imgs.view(imgs.size(0), -1)

            '''noise  = torch.rand_like(imgs)
            noisy_imgs = imgs + noise'''
            outputs = model(imgs)  # Get reconstructed images

            # Debugging prints to check shapes
            print("Input shape:", imgs.shape)
            print("Output shape:", outputs.shape)

            # Rest of the code remains the same
            imgs = imgs.view(imgs.size(0), 1, 28, 28)
            outputs = outputs.view(outputs.size(0), 1, 28, 28)
            # noisy_imgs = noisy_imgs.view(noisy_imgs.size(0), 1, 28, 28)

            for i in range(5):
                f = plt.figure()
                f.add_subplot(1, 2, 1)
                plt.imshow(imgs[i][0], cmap='gray')  # Display the original image
                '''f.add_subplot(1, 3, 2)
                plt.imshow(noisy_imgs[i][0], cmap='gray')  # Display the noisy image'''
                f.add_subplot(1, 2, 2)
                plt.imshow(outputs[i][0], cmap='gray')  # Display the denoised image
                plt.show()
            break


def interpolate_tensors(tensor1, tensor2, n_steps):
    alphas = torch.linspace(0, 1, n_steps).to(tensor1.device)  # Generate interpolation coefficients
    interpolated_tensors = []
    for alpha in alphas:
        interpolated_tensor = alpha * tensor1 + (1 - alpha) * tensor2
        interpolated_tensors.append(interpolated_tensor)
    return interpolated_tensors

def interpolate_and_visualize(model, test_loader, index1, index2, n_steps):
    model.load_state_dict(torch.load("MLP.8.pth"))
    model.eval()

    with torch.no_grad():
        for imgs, _ in test_loader:
            # Select the two images for interpolation
            img1 = imgs[index1].unsqueeze(0)  # Select image 1
            img2 = imgs[index2].unsqueeze(0)  # Select image 2

            # Encode both images to get their bottleneck tensors
            bottleneck1 = model.encode(img1)
            bottleneck2 = model.encode(img2)

            # Interpolate between the bottleneck tensors
            interpolated_tensors = interpolate_tensors(bottleneck1, bottleneck2, n_steps)

            for i, interpolated_tensor in enumerate(interpolated_tensors):
                decoded_image = model.decode(interpolated_tensor).squeeze(0)
                decoded_image = decoded_image.view(28, 28)  # Reshape to (28, 28)
                plt.subplot(1, n_steps, i + 1)
                plt.imshow(decoded_image.cpu().detach().numpy(), cmap='gray')
                plt.axis('off')

            plt.show()
            break  # Exit the loop after processing one batch of data

def main():
    print("Please enter an integer value between 0 and 59999: ")
    index = input()

    train_transform = transforms.Compose([transforms.ToTensor()])   # prepare the training data
    data_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_dataset = MNIST('./data/mnist', train=False, transform=data_transform)


    plt.imshow(train_set.data[0], cmap='gray')  # display the first element of the training set

    # Create a data loader for training data
    batch_size = 2048
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    autoencoder = autoencoderMLP4Layer()    # set up the learning model
    n_epochs = 50  # specify epochs
    learning_rate = 0.001   # adjust the learning rate
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = learning_rate)  # set up the adam optimizer
    loss_function = nn.MSELoss()    # use a mean squared loss function
    # declare the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=10, threshold=0.0001,
                                                           threshold_mode='abs')
    device = 'cpu'  # specify device type

    # train the model and return the losses_train to plot the loss function
    losses_train = train(n_epochs, optimizer, autoencoder, loss_function, train_loader, scheduler, device)
    # display the loss function
    plt.figure()
    plt.plot(losses_train)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    torch.save(autoencoder.state_dict(), "MLP.8.pth")

    summary(autoencoder, (1, 28, 28))   # print out the torch summary report

    index1 = 0  # Index of the first image to interpolate between
    index2 = 1  # Index of the second image to interpolate between
    n_steps = 10  # Number of interpolation steps
    interpolate_and_visualize(autoencoder, test_loader, index1, index2, n_steps)

    # test_reconstruction(autoencoder, test_loader, int(index))


if __name__ == '__main__':
    main()
