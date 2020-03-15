
nodes = [4, 2]
epochs = [10, 15]
batch_size = [2, 4, 6]
online_epochs = [2, 4, 6]
online_batch_size = [2, 4]
configs = list()
for i in nodes:
    for j in epochs:
        for k in batch_size:
            for l in online_epochs:
                for m in online_batch_size:
                    temp = [i, j, k, l, m]
                    configs.append(temp)
print('Total configs: %d' % len(configs))

for i in configs:
    print(i)