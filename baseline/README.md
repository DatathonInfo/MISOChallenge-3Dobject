# 객체 3D 
6D 포즈를 예측하는 모델중 [Microsoft/singleshotpose](https://github.com/microsoft/singleshotpose) 기반으로 제공합니다.
해당 알고리즘에 대한 정보는 개발 코드에 대한 github 내용을 참조하시기 바랍니다. 각 라인별로 자세한 주석을 포함하고 있습니다.

**코드의 추가/변경 내역은 [DIFF_CODE.md](https://github.com/qnqnckck/hackerton-object_3d/blob/main/baseline/DIFF_CODE.md)를 확인하시기 바랍니다.**

 데이터셋은 [LINEMOD](https://paperswithcode.com/dataset/linemod-1)를 동일한 구조를 사용합니다. LINEMOD는 6D 포즈 추정을 위한 대표적인 벤치마크로 RGB+D 데이터셋입니다. 대회에 제공될 데이터셋도 LINEMOD와 동일한 구조 변환하기 위해 데이터 전처리 과정을 필요로 합니다. 변환 방법 및 과정은 튜토리얼 내에 함께 설명합니다.

**제공되는 [객체3D데이터+정의서.pdf](https://github.com/DatathonInfo/MISOChallenge-3Dobject/files/7535380/3D.%2B.pdf)를 꼭참조하시기 바랍니다.**

***

## 1. 데이터셋 파일 경로
* 학습 데이터셋 : /mnt/hackerton/dataset/[학습데이터] (원본 + 세그멘테이션/큐브 라벨 정보를 포함)
* 배경 영상 : /mnt/hackerton/BG/JPEGImage (학습시 다양한 배경 증강을 위해 활용할 이미지를 포함)
* 개발 코드 경로: ~/[프로젝트명] 
* 데이터 정보 : ~/[프로젝트명]/data (전처리 과정시 자동생성됨) 

이미지 파일은 로컬 디스크를 복사하지 않고 NAS 내 학습데이터 이미지 사용

***
## 2. 데이터셋 전처리
### 2.1. 학습할 데이터의 경로 설정
```
line 26:
# 학습 데이터 상위 디렉토리
dataset_base_dir="/mnt/hackerton/dataset"

# 학습 데이터의 이름
data = "010304.소스용기4"

# train/test 비율 - 0.8 설정시 train : valid = 8 : 2  비율로 구성 
train_ratio = 0.8 
```

### 2.2. 데이터셋 전처리
#### 2.2.1. 제공되는 데이터셋의 폴더 구조

![스크린샷 2021-11-13 오전 1 59 38](https://user-images.githubusercontent.com/10949665/141505579-157963ed-ea41-4bd7-a406-edd3d87b135e.png)

> [데이터명].labels <br>
>> [데이터명].2D_json : origin 이미지에서 깊이(depth)정보를 제외한 단순 이미지 정보를 포함.<br>
>> [데이터명].3D_json : 2D 이미지와 더불이 깊이(depth)정보가 결합된 정보를 포함.<br>
>> [데이터명].Mask : 이미지에서 객체만을 표시한 이미지 파일.  <br>

> [데이터명].orgin <br>
>> [데이터명].3D_Shape : ply 파일을 포함하며, 데이터에 대한 속성을 가짐.  <br>
>> [데이터명].TR : 3D 이미지로 png 파일들로 구성(csv 파일은 이미지 Meta 정보).<br>

#### 2.2.2. 테스트 데이터셋 변경
```
python making_txtlables.py
```

실행 결과

![스크린샷 2021-11-11 오후 11 57 37](https://user-images.githubusercontent.com/10949665/141320742-afa5c98a-3044-443a-938f-b0ef9abe31f5.png)
      
****

## 3. 데이터셋 학습 (010304.소스용기4 예시)
### 3.1. 초기 가중치 파일과 모델 cfg 파일 다운로드
```
mkdir cfg
wget https://github.com/microsoft/singleshotpose/blob/master/cfg/yolo-pose.cfg -P cfg
wget https://pjreddie.com/media/files/darknet19_448.conv.23 -P cfg
```
### 3.2. 학습
```
python train.py \
      --datacfg data/010304.소스용기4.data \
      --modelcfg cfg/yolo-pose.cfg \
      --initweightfile cfg/darknet19_448.conv.23 \
      --pretrain_num_epochs 15
```
****

## 4. 학습된 모델 테스트 (010304.소스용기4 예시)
```    
python valid.py \
      --datacfg data/010304.소스용기4.data \
      --modelcfg cfg/yolo-pose.cfg \
      --weightfile data/010304.소스용기4/model.weights
```   
****
## 5. 결과 화면 
![result](https://user-images.githubusercontent.com/10949665/141510007-3e78fcac-4a3b-4423-a20f-047f403492b1.png)

## 6. 참고

* ref. https://github.com/microsoft/singleshotpose
      
