import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr)

a = np.shape(arr)
print(a)

print(arr[0:a[0]])

lag = 3

q_train = np.zeros((a[0]-lag, 3))
print(q_train)
for i in range(lag, a[0]):
    print("i = ", i, "arr[i] =", arr[i])
    print("i = ", i, "arr[i] =", arr[i-1])
    print("i = ", i, "arr[i] =", arr[i-2])
    q_train[i-lag][0] = arr[i]
    q_train[i-lag][1] = arr[i-1]
    q_train[i-lag][2] = arr[i-2]
    print(q_train[:][:])
    