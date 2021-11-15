[Microsoft/singleshotpose](https://github.com/microsoft/singleshotpose) 기반으로 코드 작성시 추가/변경된 파일에 대한 목록 및 변경 사항들을 정리합니다.

## 1. 추가

* **making_txtlables.py** : 학습 데이터셋에 대한 전처리를 위한 파일로 LINEMOD 데이터셋과 동일한 구조를 생성합니다.
* **valid.py** : 해당 파일의 경우 iou 연산을 위해 코드 추가/변경이 많아 대체 파일로 간주 합니다.

이미지를 제외한 데이터들을 재구성하며, 이미지는 NAS로부터 데이터를 읽습니다. 

## 2. 변경
* **dataset.py**
```
/* line 119: 데이터셋 경로 변경 */
labpath = os.path.join('data', imgpath.split('/')[len(imgpath.split('/'))-4], 'labels', imgpath.split('/')[len(imgpath.split('/'))-1].replace('.png', '.txt'))
```

* **image.py**
```
/* line 131: 데이터셋 경로 변경 */
data = imgpath.split('/')[len(imgpath.split('/')) - 4]
labpath = os.path.join('data', data, 'labels', imgpath.split('/')[len(imgpath.split('/')) - 1].replace('.png', '.txt'))
type = ".Images"
if "투명" in imgpath:
    type = ".TR"

maskpath = imgpath.replace('원천데이터', '라벨링데이터').replace(type, '.Mask').replace('.png', '_b.png')
```
 
* **Mesh.py**
```
/* line 137: 병렬처리를 위한 코드 수정 */
nProposals = int((conf > 0.25).sum().data)
```

* **train.py**
```
/* line 63: 병렬처리를 위한 코드 수정 */
seen=model.module.seen

/* line 96: 병렬처리를 위한 코드 수정 */
model.seen = model.module.seen + data.data.size(0)

/* line 145: 병렬처리를 위한 코드 수정 */
num_classes          = model.module.num_classes
anchors              = model.module.anchors
num_anchors          = model.module.num_anchors

/* line 162: 데이터셋 경로 변경 */
for batch_idx, (data, target) in enumerate(test_loader):
        with open(test_loader.dataset.lines[batch_idx].replace('원천데이터','라벨링데이터').replace('Images','3D_json').replace('_NT','_TR').replace('.png','.json').replace('\n','')) as f:
            abcde = json.load(f)
            fx = float(abcde['metaData']['Fx'])
            fy = float(abcde['metaData']['Fy'])
            u0 = float(abcde['metaData']['PPx'])
            v0 = float(abcde['metaData']['PPy'])
            internal_calibration = get_camera_intrinsic(fx,fy,u0,v0)
            
/* line 179: 병렬처리를 위한 코드 수정 */
with torch.no_grad():
  data = data
  
/* line 322: 데이터셋 경로 변경 */
bg_file_names = get_all_files('/mnt/hackerton/dataset/VOCdevkit/VOC2012/JPEGImages')

/* line 331: 제거 */
#fx          = float(data_options['fx'])
#fy          = float(data_options['fy'])
#u0          = float(data_options['u0'])
#v0          = float(data_options['v0'])

/* line 375: 제거 */
#internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

/* line 391: 병렬처리를 위한 코드 수정 */
if use_cuda:
model = torch.nn.DataParallel(model, device_ids=list(map(int, gpus.split(",")))).cuda() # Multiple GPU parallelism

/* line 422: 병렬처리를 위한 코드 수정 */
model.module.save_weights('%s/model.weights' % (backupdir))

```

* **util.py**
```
/* line 115: 좌표 연산 역배열 */
    min_x = np.min(pts[:,0])
    max_x = np.max(pts[:,0])
    min_y = np.min(pts[:,1])
    max_y = np.max(pts[:,1])
    
/* line 126: 연산식 추가 및 평가 지표를 위한 IoU 연산 함수 추가 */
def ccw(p1,p2,p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    result = ((x1 * y2 + x2 * y3 + x3 * y1)-(x2 * y1 + x3 * y2 + x1 * y3))
    return result

def get_cross_point(gt1,gt2,pr1,pr2):
    p = []
    m_gt = (gt1[1]-gt2[1])/(gt1[0]-gt2[0])
    m_pr = (pr1[1]-pr2[1])/(pr1[0]-pr2[0])
    x = ((m_gt*gt2[0])-(m_pr*pr2[0])+pr2[1]-gt2[1])/(m_gt-m_pr)
    y = m_gt*(x-gt2[0])+gt2[1]
    if gt1[0] == gt2[0]:
        x = gt1[0]
        y = (m_pr*(gt1[0]-pr2[0])+pr2[1])
    if gt1[1] == gt2[1]:
        x = ((gt1[1]-pr2[1])/m_pr)+pr2[0]
        y = gt1[1]
    p.append(x)
    p.append(y)
    return p

def compute_inner_point(gt,pr):
    inner_point = []
    n=0
    z = [0,0]
    hull_gt = spatial.ConvexHull(gt)
    hull_pr = spatial.ConvexHull(pr)
    x = hull_gt.points
    y = hull_pr.points
    u = hull_gt.vertices
    v = hull_pr.vertices
    
    for i in range(len(u)):
        for j in range(len(v)):
            if (ccw(z,x[u[i]],y[v[j-1]])*ccw(z,x[u[i]],y[v[j]])) < 0:
                if(ccw(y[v[j-1]],y[v[j]],z)*ccw(y[v[j-1]],y[v[j]],x[u[i]])) < 0:
                    n = n + 1
        if n%2 == 1:
            inner_point.append(x[u[i]][0])
            inner_point.append(x[u[i]][1])
            n=0

    for j in range(len(v)):
        for i in range(len(u)):
            if ((ccw(z,y[v[j]],x[u[i-1]])*ccw(z,y[v[j]],x[u[i]])) < 0):
                if ((ccw(x[u[i-1]],x[u[i]],z)*ccw(x[u[i-1]],x[u[i]],y[v[j]])) < 0):
                    n = n + 1
        if n%2 == 1:
            inner_point.append(y[v[j]][0])
            inner_point.append(y[v[j]][1])
            n=0

    return inner_point

def compute_cross_point(gt,pr):
    cross_point = []
    p = []
    hull_gt = spatial.ConvexHull(gt)
    hull_pr = spatial.ConvexHull(pr)
    x = hull_gt.points
    y = hull_pr.points
    u = hull_gt.vertices
    v = hull_pr.vertices
    
    for i in range(len(u)):
        for j in range(len(v)):
            if (ccw(x[u[i-1]],x[u[i]],y[v[j-1]])*ccw(x[u[i-1]],x[u[i]],y[v[j]])) < 0:
                if (ccw(y[v[j-1]],y[v[j]],x[u[i-1]])*ccw(y[v[j-1]],y[v[j]],x[u[i]])) < 0:
                    p = get_cross_point(x[u[i-1]],x[u[i]],y[v[j-1]],y[v[j]])
                    cross_point = cross_point + p

    return cross_point

def compute_convexhull_iou(gt,pr):
    cross_point = compute_cross_point(gt,pr)
    inner_point = compute_inner_point(gt,pr)
    all_inner_point = cross_point + inner_point
    if len(all_inner_point) > 5:
        all_inner_point = np.array(np.reshape(all_inner_point,[int(len(all_inner_point)/2),2]), dtype='float32')
        hull_all_inner_point = spatial.ConvexHull(all_inner_point)
        hull_gt = spatial.ConvexHull(gt)
        hull_pr = spatial.ConvexHull(pr)
        convexhull_iou = hull_all_inner_point.area / (hull_gt.area + hull_pr.area - hull_all_inner_point.area)
    else:
        convexhull_iou = 0
        
    return convexhull_iou
```
