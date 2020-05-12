import torch
import torch.nn as nn
from torchsummary import summary
torch.cuda.empty_cache()

from model import RNN, FCN, CNN
import numpy as np
import random
import time
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy 
import tqdm

input_indices = [1,5]
output_indices = [1]


n_hidden = 64
n_input = 40
n_output = 12

n_epochs = 20000
# print_every = 100
plot_every = 100
learning_rate = 0.0005 # If you set this too high, it might explode. If too low, it might not learn
l_training = 100

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomTrainingInterval():
    patient = random.choice(training_data)
    i_start = random.randint(0,len(patient) - (n_input+n_output))
    # input_data = patient[i_start:i_start+n_input,input_indices].T.flatten()
    input_data = np.expand_dims(patient[i_start:i_start+n_input,input_indices].T,axis=0)
    output_data = patient[i_start+n_input:i_start+n_input+n_output,output_indices].flatten()
    input_tensor = Variable(torch.FloatTensor(input_data))
    output_tensor = Variable(torch.FloatTensor(output_data))
    return input_data, output_data, input_tensor, output_tensor

def randomEvalInterval():
    patient = random.choice(eval_data)
    i_start = random.randint(0,len(patient) - (n_input+n_output))
    input_data = np.expand_dims(patient[i_start:i_start+n_input,input_indices].T,axis=0)
    # input_data = patient[i_start:i_start+n_input,input_indices].T.flatten()
    output_data = patient[i_start+n_input:i_start+n_input+n_output,output_indices].flatten()
    input_tensor = Variable(torch.FloatTensor(input_data))
    output_tensor = Variable(torch.FloatTensor(output_data))
    return input_data, output_data, input_tensor, output_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(input_tensor, output_tensor):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, output_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.data

def evaluate(n=1000):
    total_loss = 0
    for i in range(n):
        _,_, input_tensor, output_tensor = randomEvalInterval()
        input_tensor,  output_tensor = input_tensor.to(device), output_tensor.to(device)
        model.eval()
        output = model(input_tensor)
        loss = criterion(output, output_tensor)
        total_loss += loss
    total_loss = total_loss/n
    
    return total_loss

# Keep track of losses for plotting
current_loss = 0
all_losses_training= []
all_losses_evaluation= []

data = np.load('d1namo_insulin_rate.npz', allow_pickle=True)
data = data['data'].tolist()
training_data = data[:]
eval_data = data[:]

model = CNN(n_input,len(input_indices), n_hidden, n_output).cuda()
# model = FCN(n_input, n_hidden, n_output).cuda()
summary(model, (len(input_indices),n_input))
# print(model)

# exit()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
start = time.time()
best_eval = np.inf

for epoch in range(1, n_epochs + 1):
    input_data, output_data, input_tensor, output_tensor = randomTrainingInterval()
    input_tensor,  output_tensor = input_tensor.to(device), output_tensor.to(device)
    output, loss = train(input_tensor, output_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    # if epoch % print_every == 0:
    #     print(epoch, current_loss)

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        # scheduler.step()
        print(epoch, current_loss / plot_every)
        all_losses_training.append(current_loss / plot_every)
        eval_loss = evaluate()
        all_losses_evaluation.append(eval_loss)

        if current_loss < best_eval:
            torch.save(model.state_dict(), 'weights.pt')
            best_eval = copy.copy(current_loss)

        current_loss = 0

model.load_state_dict(torch.load('weights.pt'))
model.eval()
# plot training and validation error
plt.figure()
plt.plot(np.arange(0,len(all_losses_training))*plot_every,all_losses_training)
plt.plot(np.arange(0,len(all_losses_evaluation))*plot_every,all_losses_evaluation)
plt.xlabel('Training Iterations')
plt.ylabel('MSE Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# plot sample predictions
n_tests = 10

for i in range(n_tests):
    input_data, output_data, input_tensor, _ = randomEvalInterval()
    input_tensor = input_tensor.to(device)
    output_model = model(input_tensor)

    plt.figure()
    plt.plot(5*np.arange(0,n_input), input_data[0,0,:n_input], 'k--')
    plt.plot(5*np.arange(n_input,n_input+n_output), output_data, 'r')
    plt.plot(5*np.arange(n_input,n_input+n_output), output_model.cpu().detach().numpy(), 'g')
    plt.legend(['Input Glucose', 'Ground Truth Output', 'Predicted Output'])
    plt.xlabel('Time (mins)')
    plt.ylabel('Glucose level mmol/l')
    plt.show()


