import json
import os
import random
import math
import shutil
from MeshPly import MeshPly
from utils import *

'''##############################################################################################
# DESCRIPTION
 데이터셋의 필요한 속성은 실행한 폴더를 기준으로 ./data 경로에 파일들을 생성하며, 이미지 파일은 용량이 크기 때문에 NAS /dataset의 하위 경로에서 읽어오도록 한다.

<OPTIONS>
dataset_base_dir : 객체 3D 데이터 경로(라벨링데이터, 원천데이터 폴더를 포함)]
data : 객체 3D 대상명
train_ratio : train/test dataset의 비율


<OUTPUT>
LINEMOD 데이터 구성을 위해, 다음 파일들을 생성합니다.
- test.txt
- train,txt
- [폴더명].data
##############################################################################################'''

dataset_base_dir="/mnt/hackerton/dataset"
data = "010304.source4"

# ratio
train_ratio = 0.8  # test_ratio = 1 -train_ratio

# 원천데이터
origin_data_dir = dataset_base_dir + "/" + data + "/" + data + ".origin"
origin_image_dir = origin_data_dir + "/" + data + ".Images"
origin_threed_shape_data_dir = origin_data_dir + "/" + data + ".3D_Shape"

# 라벨링데이터
labeling_dir = dataset_base_dir + "/" + data + "/" + data + ".labels"
labeling_threed_json_dir = labeling_dir + "/" + data + ".3D_json"

##################################
# data 파일 생성
'''
train  = LINEMOD/ape/train.txt
valid  = LINEMOD/ape/test.txt
backup = backup/ape
mesh = LINEMOD/ape/ape.ply
tr_range = LINEMOD/ape/training_range.txt
name = ape
diam = 0.103
gpus = 0
width = 640
height = 480
fx = 572.4114 
fy = 573.5704
u0 = 325.2611
v0 = 242.0489
'''
if not (os.path.isdir(os.path.join('cfg'))):
    os.makedirs(os.path.join('cfg'))

class DataProperty:
    def setTrain(self, train):
        self.train = train

    def setValid(self, valid):
        self.valid = valid

    def setMesh(self, mesh):
        self.mesh = mesh

    def setBackup(self, backup):
        self.backup = backup

    def setTrRange(self, tr_range):
        self.tr_range = tr_range

    def setName(self, name):
        self.name = name

    def setDiam(self, diam):
        self.diam = diam

    def setGpus(self, gpus):
        self.gpus = gpus

    def setWidth(self, width):
        self.width = width

    def getWidth(self):
        return self.width

    def setHeight(self, height):
        self.height = height

    def getHeight(self):
        return self.height

    def setFx(self, fx):
        self.fx = fx

    def setFy(self, fy):
        self.fy = fy

    def setU0(self, u0):
        self.u0 = u0

    def setV0(self, v0):
        self.v0 = v0

    def toString(self):
        ret = 'train = ' + self.train + '\n'
        ret += 'valid = ' + self.valid + '\n'
        ret += 'mesh = ' + self.mesh + '\n'
        ret += 'backup = ' + self.backup + '\n'
        ret += 'tr_range = ' + self.tr_range + '\n'
        ret += 'name = ' + self.name + '\n'
        ret += 'diam = ' + str(self.diam) + '\n'
        ret += 'gpus = ' + str(self.gpus) + '\n'
        ret += 'width = ' + self.width + '\n'
        ret += 'height = ' + self.height + '\n'
        ret += 'fx = ' + self.fx + '\n'
        ret += 'fy = ' + self.fy + '\n'
        ret += 'u0 = ' + self.u0 + '\n'
        ret += 'v0 = ' + self.v0

        return ret


if not (os.path.isdir("data")):
    os.makedirs(os.path.join("data"))

if not (os.path.isdir(os.path.join("data",data))):
    os.makedirs(os.path.join("data",data))
data_file = open('data/' + data  + '.data', 'w')

dataProperty = DataProperty();
dataProperty.setTrain('data/' + data + '/train.txt')
dataProperty.setValid('data/' + data + '/test.txt')
dataProperty.setMesh('data/' + data + '/' + data.split('.')[0] + '.ply')
dataProperty.setBackup('backup/' + data )
dataProperty.setTrRange('data/' + data + '/training_range.txt')
dataProperty.setName(data.split('.')[1])

## ply파일 복사
ply_filename = data.split('.')[0] + '.ply'
shutil.copy(origin_threed_shape_data_dir + "/" + ply_filename, "data/" + data + "/" + ply_filename)

mesh = MeshPly("data/" + data + "/" + ply_filename)
diam = calc_pts_diameter(np.array(mesh.vertices))

dataProperty.setDiam(diam)

dataProperty.setGpus("0,1")

name_list = os.listdir(labeling_threed_json_dir)

with open(labeling_threed_json_dir + '/' + name_list[0], 'r') as f:
    data2 = json.load(f)
    dataProperty.setWidth(data2['metaData']['Resolution x'])
    dataProperty.setHeight(data2['metaData']['Resolution y'])
    dataProperty.setFx(data2['metaData']['Fx'])
    dataProperty.setFy(data2['metaData']['Fy'])
    dataProperty.setU0(data2['metaData']['PPx'])
    dataProperty.setV0(data2['metaData']['PPy'])

with open(os.path.join('cfg', data + ".data"), 'w') as file:
    file.write(dataProperty.toString())

##################################

name_list = os.listdir(labeling_threed_json_dir)

labels_txt_data_dir = os.path.join('data', data, 'labels');
if not (os.path.isdir(labels_txt_data_dir)):
    os.makedirs(os.path.join(labels_txt_data_dir))


width = float(dataProperty.getWidth())
height = float(dataProperty.getHeight())

for i in name_list:
    with open(labeling_threed_json_dir + '/' + i, 'r') as f:
        data2 = json.load(f)

        location = data2['labelingInfo'][0]['3DBox']['location'][0]
        k = open(labels_txt_data_dir + '/' + i[-21:-5] + '.txt', 'w')
        k.write('3 ')
        k.write(str(float(location['x9'])/width))
        k.write(' ')
        k.write(str(float(location['y9'])/height))
        k.write(' ')
        k.write(str(float(location['x4'])/width))
        k.write(' ')
        k.write(str(float(location['y4'])/height))
        k.write(' ')
        k.write(str(float(location['x1'])/width))
        k.write(' ')
        k.write(str(float(location['y1'])/height))
        k.write(' ')
        k.write(str(float(location['x8'])/width))
        k.write(' ')
        k.write(str(float(location['y8'])/height))
        k.write(' ')
        k.write(str(float(location['x5'])/width))
        k.write(' ')
        k.write(str(float(location['y5'])/height))
        k.write(' ')
        k.write(str(float(location['x3'])/width))
        k.write(' ')
        k.write(str(float(location['y3'])/height))
        k.write(' ')
        k.write(str(float(location['x2'])/width))
        k.write(' ')
        k.write(str(float(location['y2'])/height))
        k.write(' ')
        k.write(str(float(location['x7'])/width))
        k.write(' ')
        k.write(str(float(location['y7'])/height))
        k.write(' ')
        k.write(str(float(location['x6'])/width))
        k.write(' ')
        k.write(str(float(location['y6'])/height))
        k.write(' ')
        k.write(str(float(location['x-range'])/width))
        k.write(' ')
        k.write(str(float(location['y-range'])/height))
        k.write(' ')

data_root = os.path.join('data', data, 'labels')
loader_root = os.path.join('data', data)

name_list2 = os.listdir(data_root)
image_list = [name for name in name_list2 if name[-4:] == '.txt']
train_len = math.floor(len(image_list) * train_ratio)

train_name = random.sample(image_list, train_len)
valid_name = [name for name in image_list if name not in train_name]

with open(os.path.join(loader_root, 'train.txt'), 'w') as file:
    for i in range(len(train_name)):
        file.write(os.path.join(origin_image_dir, train_name[i].replace('.txt', '.png') + '\n'))

with open(os.path.join(loader_root, 'test.txt'), 'w') as file:
    for i in range(len(valid_name)):
        file.write(os.path.join(origin_image_dir, valid_name[i].replace('.txt', '.png') + '\n'))

with open(os.path.join(loader_root, 'training_range.txt'), 'w') as file:
    for i in range(len(train_name)):
        file.write(str(int(train_name[i].replace('.txt', '').split('_')[1])) + '\n')
