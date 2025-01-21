import os

def make_datalist(data_fd, data_list, gt_fd, data_gt_list):

    filename_all = os.listdir(data_fd)
    filename_all = [data_fd + img_name + '\n' for img_name in filename_all if img_name.endswith('.npy')]
    
    print(filename_all[0])
    print(len(filename_all))

    gt_filename_all = []
    for img_name in filename_all:
        gt_id = os.path.splitext(img_name.split('/')[-1])[0] + '_gt.npy'
        gt_filename_all.append(gt_fd + gt_id + '\n')

    print(len(gt_filename_all))
    
    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)

    with open(data_gt_list, 'w') as fp:
        fp.writelines(gt_filename_all)

if __name__ == '__main__':
    data_fd = './data/data_np/train_ct/'
    gt_fd = './data/data_np/gt_train_ct/'
    data_list = './data/datalist/train_ct.txt'
    data_gt_list = './data/datalist/train_ct_gt.txt'

    make_datalist(data_fd, data_list, gt_fd, data_gt_list)
        