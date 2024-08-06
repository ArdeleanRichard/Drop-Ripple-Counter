import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff
from ucimlrepo import fetch_ucirepo

seed = 30
random_state = 170

def create_dataX(avgPoints=250):
    np.random.seed(0)
    C4 = [0, 0] + .4 * np.random.randn(avgPoints // 5, 2)
    C5 = [0, -3] + .7 * np.random.randn(avgPoints * 5, 2)
    C3 = [2, 5] + .5 * np.random.randn(avgPoints * 4, 2)
    C6 = [-2, 5] + 1.0 * np.random.randn(avgPoints * 3, 2)

    X = np.vstack((C5, C3, C4, C6))

    c3Labels = np.full(len(C3), 0)
    c4Labels = np.full(len(C4), 1)
    c5Labels = np.full(len(C5), 2)
    c6Labels = np.full(len(C6), 3)

    # 1: 'red',
    # 2: 'blue',
    # 3: 'green',
    # 4: 'black',
    # 5: 'yellow',
    # 6: 'cyan',

    y = np.hstack((c5Labels, c3Labels, c4Labels, c6Labels))
    return (X, y)

def create_set(n_samples):
    dataX = create_dataX()
    # data1 = create_data1(n_samples)
    # data2 = create_data2(n_samples)
    data3 = create_data3(n_samples)
    data4 = create_data4(n_samples)
    data5 = create_data5(n_samples)
    data6 = create_data6(n_samples)
    data7 = create_data7(n_samples)
    data8 = create_data8()
    data9 = create_data9()
    # data10 = create_data10()
    data11 = create_data11()

    datasets = [
        # data1,
        # data2,
        dataX,
        data3,
        data4,
        data5,
        data6,
        data7,
        data8,
        data9,
        data11
    ]


    return datasets


def create_set1(n_samples):
    # data1 = create_data1(n_samples)
    # data2 = create_data2(n_samples)
    data3 = create_data3(n_samples)
    data4 = create_data4(n_samples)
    data5 = create_data5(n_samples)
    data6 = create_data6(n_samples)
    data7 = create_data7(n_samples)

    datasets = [
        # data1,
        # data2,
        data3,
        data4,
        data5,
        data6,
        data7,
    ]

    return datasets


def create_set2():
    data8 = create_data8()
    data9 = create_data9()
    data10 = create_data10()
    data11 = create_data11()

    datasets = [
        data8,
        data9,
        data10,
        data11
    ]

    return datasets


def create_data1(n_samples):
    avgPoints = n_samples // 3
    C1 = [-5, -10] + .8 * np.random.randn(avgPoints, 2)
    C2 = [5, -10] + .8 * np.random.randn(avgPoints, 2)
    C3 = [5, 10] + .8 * np.random.randn(avgPoints, 2)

    X = np.vstack((C1, C2, C3))

    c1Labels = np.full(len(C1), 0)
    c2Labels = np.full(len(C2), 1)
    c3Labels = np.full(len(C3), 2)

    y = np.hstack((c1Labels, c2Labels, c3Labels))

    data1 = (X, y)

    return data1

def create_data2(n_samples):
    avgPoints = n_samples // 5
    C1 = [5, -10] + .8 * np.random.randn(avgPoints, 2)
    C2 = [0, -9] + .8 * np.random.randn(avgPoints, 2)
    C3 = [-5, -5] + .8 * np.random.randn(avgPoints, 2)
    C4 = [1, 0] + .8 * np.random.randn(avgPoints, 2)
    C5 = [8, -1] + .8 * np.random.randn(avgPoints, 2)

    X = np.vstack((C1, C2, C3, C4, C5))

    c1Labels = np.full(len(C1), 0)
    c2Labels = np.full(len(C2), 1)
    c3Labels = np.full(len(C3), 2)
    c4Labels = np.full(len(C4), 3)
    c5Labels = np.full(len(C5), 4)

    y = np.hstack((c1Labels, c2Labels, c3Labels, c4Labels, c5Labels))

    data2 = (X, y)

    return data2



def create_data3(n_samples):
    return datasets.make_blobs(n_samples=n_samples, random_state=seed)


def create_data4(n_samples):
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=1.0, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    return aniso

def create_data5(n_samples, n_features=2):
    # data5 with data3 variances
    return datasets.make_blobs(n_samples=n_samples, n_features=n_features, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)


def create_data6(n_samples):
    return datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)


def create_data7(n_samples):
    return datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)


def create_data8():
    data, _ = arff.loadarff('./datasets/3-spiral.arff')
    data = transform_arff_data(data)

    return data


def create_data9():
    data = pd.read_csv('./datasets/unbalance.csv', header=None)
    temp_data = data.to_numpy()
    data = (temp_data[:, :-1], temp_data[:, -1])
    return data

def create_data10():
    data, _ = arff.loadarff('./datasets/aggregation.arff')
    data = transform_arff_data(data)
    return data

def create_data11():
    data = pd.read_csv('./datasets/s1_labeled.csv', header=None)
    temp_data = data.to_numpy()
    data = (temp_data[:, :-1], temp_data[:, -1])
    return data



def create_data12(n_samples):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, 2)
    no_structure = (X, np.zeros((len(X))))
    return no_structure





def create_data3_3d(n_samples):
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=3, cluster_std=1.0, random_state=random_state)
    transformation = [[0.3, -0.3, 0.01], [-0.2, 0.4, 0.01], [-0.1, 0.2, 0.01]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    return aniso

def create_data4_3d(n_samples):
    return datasets.make_blobs(n_samples=n_samples, n_features=3, random_state=seed)


def create_data5_3d(n_samples):
    # data5 with data3 variances
    return datasets.make_blobs(n_samples=n_samples, n_features=3, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)


def create_set3d(n_samples):
    data3 = create_data3_3d(n_samples)
    data4 = create_data4_3d(n_samples)
    data5 = create_data5_3d(n_samples)

    datasets = [
        data3,
        data4,
        data5,
    ]

    return datasets




def transform_arff_data(data):
    X = []
    y = []
    for sample in data:
        x = []
        for id, value in enumerate(sample):
            if id == len(sample) - 1:
                y.append(value)
            else:
                x.append(value)
        X.append(x)


    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)


def read_uci(fetched_data):
    X = fetched_data.data.features.to_numpy()
    y = fetched_data.data.targets.to_numpy().squeeze()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y

def create_ecoli():
    # data, meta = arff.loadarff('./data/ecoli.arff')
    # return transform_arff_data(data)

    fetched_data = fetch_ucirepo(id=39)
    return read_uci(fetched_data)

def create_glass():
    # data, meta = arff.loadarff('./data/glass.arff')
    # return transform_arff_data(data)

    fetched_data = fetch_ucirepo(id=42)
    return read_uci(fetched_data)


def create_yeast():
    # data, meta = arff.loadarff('./data/yeast.arff')
    # return transform_arff_data(data)

    fetched_data = fetch_ucirepo(id=110)
    return read_uci(fetched_data)


def create_statlog():
    fetched_data = fetch_ucirepo(id=147)
    return read_uci(fetched_data)

def create_wdbc():
    fetched_data = fetch_ucirepo(id=17)
    return read_uci(fetched_data)


def create_wine():
    fetched_data = fetch_ucirepo(id=109)
    return read_uci(fetched_data)


def create_sonar():
    fetched_data = fetch_ucirepo(id=151)
    return read_uci(fetched_data)

def create_ionosphere():
    fetched_data = fetch_ucirepo(id=52)
    return read_uci(fetched_data)

def create_setHD():
    data1 = create_ecoli()
    data2 = create_glass()
    data3 = create_yeast()
    data4 = create_statlog()
    data5 = create_wdbc()
    data6 = create_wine()
    data7 = create_sonar()
    data8 = create_ionosphere()

    datasets = [
        data1,
        data2,
        data3,
        data4,
        # data5,
        # data6,
        data7,
        # data8,
    ]

    return datasets




