import numpy as np

def MakePattern2d(shape='o', n_points=10, scale=(1.0, 1.0), sgma=(1.0, 1.0), x0=(1.0, 1.0)):
    '''
    Generate 2D patterns with explicit scaling and noise steps
    
    Parameters:
    - shape: Pattern type ('o', 'c', 'b', 'x')
    - n_points: Number of sample points
    - sgma_x, sgma_y: Scaling factors for axes
    - x0, y0: Pattern center coordinates
    '''
    
    outp = np.zeros((n_points, 2))
    
    # Noise:
    noise_x = np.random.normal(0, 1, n_points)*sgma[0]
    noise_y = np.random.normal(0, 1, n_points)*sgma[1]

    # Base pattern generation
    if shape == 'b':  # Blob (base: standard normal)
        
        pass
        
    elif shape == 'o':  # Ring (unit circle)

        theta = np.random.uniform(0, 2*np.pi, n_points)
        outp[:, 0] = np.cos(theta)
        outp[:, 1] = np.sin(theta)
        

    elif shape == 'c':  # Crescent (240° arc)
        dth = (n_points // 16) * np.pi/8
        theta = np.random.uniform(np.pi/3, 5*np.pi/3, n_points) + dth
        outp[:, 0] = np.cos(theta)
        outp[:, 1] = np.sin(theta)

        
    elif shape == 'x':  # X-shape (standard normal)
        t = np.random.normal(0, 1, n_points)
        slope = np.random.choice([1, -1], n_points)
        outp[:, 0] = t
        outp[:, 1]= slope * t
        
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # Scaling
    if shape != 'b':
        outp[:, 0] *= scale[0]
        outp[:, 1] *= scale[1]

    # Centering
    outp[:, 0] += x0[0]
    outp[:, 1] += x0[1]

    # Noise addition (except for blob)
    outp[:, 0] += noise_x
    outp[:, 1] += noise_y

    return outp


def split_data(X, y, test_size=0.2, random_state=42):
    
    # YOUR CODE
    np.random.seed(random_state)

    indices = np.arange(len(X))

    np.random.shuffle(indices)

    split_point = int(len(X) * (1 - test_size))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


if __name__=='__main__':

    import matplotlib.pyplot as plt

    o_sample = MakePattern2d('o', 100, (2.0, 1.0), (0.2, 0.2), (-2.0, 2.0))
    c_sample = MakePattern2d('c', 102, (0.5, -1.0), (0.1, 0.1), (2.0, 2.0))
    b_sample = MakePattern2d('b', 100, (1.0, 1.0), (0.5, 1.0), (-2.0, -2.0))
    x_sample = MakePattern2d('x', 100, (1.0, 1.5), (0.4, 0.2), (2.0, -2.0))

    fig, ax = plt.subplots(1,1,figsize=(8,8))

    ax.scatter(o_sample[:, 0], o_sample[:, 1], c='r')
    ax.scatter(c_sample[:, 0], c_sample[:, 1], c='m')
    ax.scatter(b_sample[:, 0], b_sample[:, 1], c='b')
    ax.scatter(x_sample[:, 0], x_sample[:, 1], c='y')

    ax.legend(['o', 'c', 'b', 'x'])
    ax.grid('on')

    plt.show()
