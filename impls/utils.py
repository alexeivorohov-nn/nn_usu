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


class BatchSampler:

    def __init__(self, X_train, y_train, batch_size, n_batches):

        self.X_train = X_train
        self.y_train = y_train
        self.x_size = X_train.shape[0]

        self.batch_size = batch_size
        self.n_batches = n_batches
        
        # Validate input constraints
        if self.batch_size * self.n_batches > self.x_size:
            raise ValueError(
                f"Product of batch_size ({batch_size}) and n_batches ({n_batches}) "
                f"exceeds dataset size {self.x_size}. Cannot sample without repetition."
            )

    def sample(self):
        
        # Generate single permutation for all batches
        full_perm = torch.randperm(self.x_size)
        total_samples = self.batch_size * self.n_batches
        
        # Select contiguous indices from permutation
        selected_indices = full_perm[:total_samples]
        
        # Create non-overlapping batches
        batches = []
        for i in range(self.n_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_indices = selected_indices[start:end]
            
            batches.append(
                (self.X_train[batch_indices], self.y_train[batch_indices])
            )
        
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
    
    def fit_transform(self, x):

        self.fit(x)
        return self.transform(x)
    
    def inverse(self, x):
        # YOUR_CODE
        x_scaled = x*self.x_std + self.x_mean

        return x_scaled
    

if __name__ == '__main__':

    A = torch.randn((8, 2))

    loader = TimeseriesLoader(A, 2, 2, 1)

    for d in loader:
        print(d)


    pass

