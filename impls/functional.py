import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_loop(model, loss, sampler, lr=0.002, n_epochs=30):
    
    loss_history = []
    
    y_eval = model.forward(sampler.X_train)
    loss_history.append(loss(y_eval, sampler.y_train))

    for epoch in range(n_epochs):

        batches = sampler.sample()

        for (x_b, y_b) in batches:

            y_pred = model.forward(x_b)
            # loss_val = loss(y_pred, y_b) - уже не нужно вызывать
            loss_grad = loss.grad(y_pred, y_b)

            model_grad_w = model.grad_w(x_b, loss_grad)
            model_grad_b = model.grad_b(x_b, loss_grad)

            model.w -= lr*model_grad_w
            model.b -= lr*model_grad_b

        
        y_eval = model.forward(sampler.X_train)
        loss_history.append(loss(y_eval, sampler.y_train))


    # Визуализация
    fig, ax0 = plt.subplots(1,1, figsize=(5, 5))


    epoch_span = torch.arange(0, n_epochs+1, 1)

    ax0.plot(epoch_span, loss_history, 'r-', label='Loss')
    ax0.set_xlim(0, n_epochs)
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    ax0.set_title('Loss Over Epochs')
    ax0.legend()

    plt.show()

    pass