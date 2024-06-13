import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

#Import Dataset
def dataset(n):
    """"
    Fungsi untuk Preprocessing Data Transien
    Dataset harus diletakan pada folder yang sama dengan file ini dalam folder sendiri bernama "Data Training"
    Nama file input memiliki format n.csv yang merupakan output default dari Simulator iPWR Technatom
    dengan n merupakan suatu bilangan bulat yang menunjukan urutan dataset.
    Fungsi ini mengembalikan suatu numpy array dengan dimensi [Jml Transien, Jml Timestep, Jml Sinyal]
    """
    path = f'./Data Training/{n}.csv'
    arr = np.genfromtxt(path, delimiter=';', skip_header=1)
    arr_fixed = np.delete(arr, np.arange(1, len(arr[0]), 2), axis=1)[:,:-1]
    return arr_fixed

#Main Function
def main():
    N = 50
    filterSignal = np.array([2, 10, 11, 12, 18, 20, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 64, 66, 67, 69, 71, 72, 74, 75, 83, 87, 88, 93, 94, 95, 98, 106])
    X_train = np.array([dataset(i) for i in range(1,N+1)])[:,270:,filterSignal+1]

    seed = 0
    np.random.seed(seed)
    Sum_of_squared_distances = []
    K = range(1,8)
    for k in K:
        km = TimeSeriesKMeans(n_clusters=k,
                              n_init=2,
                              metric="dtw",
                              verbose=True,
                              max_iter_barycenter=10,
                              random_state=seed)
        km = km.fit(X_train)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('Jumlah Klaster, K')
    plt.ylabel('Jumlah Jarak Sampel ke Pusat Klaster Terdekat, $\sigma$')
    plt.title('Elbow Method')
    plt.savefig('elbowMethod.png')
    plt.show()
    return 0

#Call Main Function
main()

