import torch
import torch.nn as nn

class TimeseriesLoader():

    def __init__(self, data: torch.Tensor, lookback, fwd_scope=1, stride=1):

        self.N_points = data.shape[0]
        self.data = data

        self.lookback = lookback # Число прошлых точек данных, которые используем для предсказания
        self.fwd_points = fwd_scope
        self.stride = stride        

        return None


    def __iter__(self):

        self.N0 = 0
        self.N1 = self.lookback
        self.N2 = self.lookback + self.fwd_points

        stride = self.stride

        self.next_id = lambda N0, N1, N2 : (N0 + stride, N1 + stride, N2 + stride)
        self.last_N = self.N_points - self.fwd_points

        return self


    def __next__(self):

        if self.N2 < self.N_points:
            x = self.data[self.N0:self.N1]
            y = self.data[self.N1:self.N2]

            self.N0, self.N1, self.N2 = self.next_id(self.N0, self.N1, self.N2)

            return (x, y)

        else:
            raise StopIteration


if __name__ == '__main__':

    A = torch.randn((8, 2))

    loader = TimeseriesLoader(A, 2, 2, 1)

    for d in loader:
        print(d)


    pass


class BatchSampler:

    def __init__(self, X_train, y_train, batch_size, n_batches):

        self.X_train = X_train
        self.y_train = y_train
        self.x_size = X_train.shape[0]

        self.batch_size = batch_size
        self.n_batches = n_batches

        pass

    def sample(self):

        batches = []
        for i in range(self.n_batches):
            perm = torch.randperm(self.x_size)[0:self.batch_size]

            batches.append((self.X_train[perm], self.y_train[perm]))

        return batches
    

class StandardScaler:

    def __init__(self):
        pass

    def fit(self, x):
        # YOUR_CODE
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)

        pass
    
    def transform(self, x):
        # YOUR_CODE

        x_scaled = (x-self.x_mean)/self.x_std

        return x_scaled
    
    def inverse(self, x):
        # YOUR_CODE
        x_scaled = x*self.x_std + self.x_mean

        return x_scaled