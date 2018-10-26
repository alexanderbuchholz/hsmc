import numpy as np
@profile
def test_function(x):
    x.sum()

if __name__ == '__main__':
    x = np.random.normal(size=(1000000))
    test_function(x)
