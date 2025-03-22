import torch

# Место для реализации вашей коллекции loss-функций

def mock_loss_2d(w):

    x = w[..., 0]
    y = w[..., 1]

    s_y = torch.sigmoid(10*y)

    borders = torch.sigmoid(4*(abs(x)-3)) + torch.sigmoid(4*(abs(y)-3))
    plato1 = torch.sigmoid(x+0.7*y)
    minima = -torch.exp(-torch.cosh((x-2)**2 + (y+2)**2))
    plato2 = -torch.sigmoid(x)-torch.exp(-(y+2)**2)
    osc = torch.cos(torch.pi*(x-1))*0.1+torch.sin(torch.pi*(y-1))*0.06
    outp = borders + (minima+plato2)*(1-s_y) + plato1*s_y + osc

    return torch.tanh(outp/10)