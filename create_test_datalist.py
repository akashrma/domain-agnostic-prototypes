import os

def make_datalist(data_fd, data_list):
 
    filename_all = os.listdir(data_fd)
    filename_all = [data_fd + img_name + '\n' for img_name in filename_all if img_name.endswith('.npz')]

    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)

if __name__ == '__main__':

    data_fd = './data/data_np/test_mr/'
    data_list = './data/datalist/test_mr.txt'

    make_datalist(data_fd, data_list)
        