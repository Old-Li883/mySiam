import json
import os

def file_name(file_dir):
    L = []
    # root,dirs,_ = os.walk(file_dir)
    for root, dirs, files in os.walk(file_dir):
        # for file in files:

        #     if os.path.splitext(file)[1] == '.csv':
        #         L.append(os.path.join(root, file))
        #     elif os.path.splitext(file)[1] == '.xlsx':
        for d in dirs:
            L.append(os.path.join(root, d))
        break
    return L


def total_file(l):
    count = 0
    for path in l:
        count += len(os.listdir(path))

    return count



if __name__ == "__main__":
    # labels = json.load(open('/mnt3/lichenhao/VISO/train2017.json', 'r'))  # 双字典形式
    # keyName = list(labels.keys())
    # targetList = []
    # for k in keyName:
    #     targetList.extend(list(labels[k].keys()))

    # print('target num', len(targetList))
    l = file_name(r'/mnt3/lichenhao/VISO/crop99')
    count = total_file(l)
    print(count)

    print(len(os.listdir(r'/mnt3/lichenhao/VISO/crop99/001')))