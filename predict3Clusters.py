from scipy import interpolate

inputFile = 'tl30' #change this variable for other transien testing

def testDataset(n):
  path = f'./Data Uji Coba_2/{n}.csv'
  arr = np.genfromtxt(path, delimiter=';', skip_header=1)
  arr_fixed = np.delete(arr, np.arange(1, len(arr[0]), 2), axis=1)[:,:-1]
  t = arr_fixed[:, 0]
  t0 = 0
  tn = 601
  transient = np.zeros(shape=(tn, arr_fixed.shape[-1]))

  for i in range(transient.shape[-1]):
    interpolator = interpolate.interp1d(t, arr_fixed[:,i])
    transient[:, i] = interpolator(np.arange(t0, tn))
  return transient

# Import data 1.csv-50.csv to a variable X
X_test = np.array([testDataset(inputFile)[270:,filterSignal+1]])
pred = km.predict(X_test)
print(pred)
T = np.array([dataset(i) for i in range(1,N+1)])[:,270:, 0]

fig, ax = plt.subplots(X_train.shape[2], C)
fig.set_size_inches(10, 100)

for z in range(X_train.shape[2]):
    for i, X in enumerate(X_train):
        line1, = ax[z, y_pred[i]].plot(T[i,:],
                              X[:,z], 
                              color='k', 
                              alpha=.4,
                              label='Training Data',
                              linewidth=1)

    line2, = ax[z, pred[0]].plot(T[i,:], 
                                 X_test[0, :, z], 
                                 color='b', 
                                 label=f'Transient {inputFile}', 
                                 linewidth=1)
    ax[z, 0].set_ylabel(filteredSignal[z])
    ax[z, 1].set_yticklabels([])
    ax[z, 2].set_yticklabels([])
    ax[z, 1].set_xlabel("Simulation Time (second)")

    for j in range(C):
        ax[z, j].set_ylim((np.min(X_train[:,:,z]), np.max(X_train[:,:,z])))
        #ax[0, j].set_title(f"Cluster {j+1}")
        ax[z, j].set_title(f"Cluster {j+1}")
        if j==C-1:
            ax[z, j].legend(handles=[line1, line2], fontsize=7)

plt.tight_layout()
fig.savefig(f'Hasil Uji Coba Transien {inputFile}.png')
plt.show()
