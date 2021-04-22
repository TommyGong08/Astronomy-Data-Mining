import os

path = './dataset/first_train_data/first_train_data/'
with open("./data_list.txt",'w') as f:
    for item in os.listdir(path):
        file_name = item.split('.')[0]
        print(file_name)
        f.writelines(file_name+'\n')

    f.close()
