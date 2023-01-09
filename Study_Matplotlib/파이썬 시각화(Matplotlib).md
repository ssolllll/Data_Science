# 파이썬 시각화(Matplotlib)

### plt.figure(figsize Argument)

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(7,7))
plt.show()
```

(이하, import 패키지 생략)

![Matplotlib_0](jpg/Matplotlib_0.png)

### plt.figure(Other Argument)

```python
fig = plt.figure(figsize=(7,7),
                 facecolor='darkgreen')
plt.show()
```

![Matplotlib_1](jpg/Matplotlib_1.png)

### fig.add_subplot(Adding Subplots)

```python
fig = plt.figure(figsize=(7,7),
                 facecolor='darkgreen')
ax = fig.add_subplot()
plt.show()
```

!Matplotlib_2](jpg/Matplotlib_2.png)

### fig.add_subplot(Adding Subplots)

```python
fig = plt.figure(figsize = (7,7),
                 facecolor='darkgreen')
ax = fig.add_subplot()
ax.scatter([2,3,1],[2,3,4])
plt.show()
```

 

![Matplotlib_3](jpg/Matplotlib_3.png)

### fig.subptitle and ax.set_title(Title of Figure)

```python
figsize = (7,7)
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Title of a Figure',
             fontsize=30,
             fontfamily='sans')
plt.show()
```

!Matplotlib_4](jpg/Matplotlib_4.png)

### ax.grid()

```python
fig,ax = plt.subplots(figsize=(7,7))
ax.grid()
plt.show()
```

- 격자 모양 활성화

![Matplotlib_5](jpg/Matplotlib_5.png)

### ax.set_xlabel() & ax.set_ylabel()

```python
fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlabel('X label',
              fontsize=20)
ax.set_ylabel('Y label',
              fontsize=20,
              alpha = 0.7)
plt.show()
```

- X & Y 축 타이틀 설정
- alpha
    - 투명도 설정
    - default = 1
    - float value (0 ~ 1 사이의 값)

![Matplotlib_6](jpg/Matplotlib_6.png)

### fig.tight_layout()

- 자동으로 여백(padding)에 관련된 서브플롯 파라미터 조정.
- default value는 자동으로 레이아웃 설정.

### ax.twinx() & ax.set_xlim

```python
fig = plt.figure(figsize=(10,10),
                 facecolor='yellowgreen')
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
ax1.set_xlim([0,100])
ax1.set_ylim([0,100])
ax2.set_ylim([0,10])

plt.show()
```

![Matplotlib_7](jpg/Matplotlib_7.png)

### ax.tick_params()

- 눈금 설정

| parameter | option | description |
| --- | --- | --- |
| axis | default 
parameter
description | both
’x’ , ‘y’, ‘both’ |
| labelsize | default 
parameter 
description | none
float or str
눈금 폰트 크기 float 형태로 지정 or 문자열로 가능(e.g., ‘large’) |
| length | default
parameter
description | none
float
 |
| width | default
parameter
description | none
float |
| rotation | default
parameter
description | none
float ( 글자 회전각이므로 0~ 360 정도로 판단)
글자 겹치는 걸 방지하기 위해 회전. |
| which | default
parameter
description | major
’major’ , ‘minor’ , ‘both’
설정할 눈금을 어디에 적용할 지 설정. |
- ax.tick_params(labelsize = 20)

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
ax.tick_params(labelsize=20,
               length=10)
plt.show()
```

![Matplotlib_8](jpg/Matplotlib_8.png)

- default

![- labelsize = 20](jpg/Matplotlib_9.png)

- labelsize = 20

![- length = 10](jpg/Matplotlib_10.png)

- length = 10

### ****ax.tick_params(Tick Locations)****

- bottom, left, right, top 설정 가능

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
ax.tick_params(bottom=False,
               labelbottom=False)
plt.show()
```

![- default](jpg/Matplotlib_11.png)

- default

![bottom = False](jpg/Matplotlib_12.png)

bottom = False

![bottom = False, labelbottom=False](jpg/Matplotlib_13.png)

bottom = False, labelbottom=False

### ax.tick_params(rotation Arguments)

- rotation > 0 : 왼쪽으로 기울기
- rotation < 0 : 오른쪽으로 기울기

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
ax.tick_params(rotation = 30) # - 360 < 0 < 360 까지 가능.
plt.show()
```

![- default](jpg/Matplotlib_11.png)

- default

![- rotation = 30](jpg/Matplotlib_14.png)

- rotation = 30

![- rotation = -30](jpg/Matplotlib_15.png)

- rotation = -30

### ax.tick_params(x, y Axis Ticks)

- axis = [’x’ , ‘y’ , ‘both’] 를 활용하여 한 축만 변경할 수 있음.
- default값은 ‘both’

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
ax.tick_params(axis='x',
               rotation=20)
plt.show()
```

![- default](jpg/Matplotlib_11.png)

- default

![- axis = ‘x’](jpg/Matplotlib_16.png)

- axis = ‘x’

![- axis = ‘y’](jpg/Matplotlib_17.png)

- axis = ‘y’

### ax.text()

- (x,y) 좌표에 텍스트 입력

| parameter | option | description |
| --- | --- | --- |
| x,y | default 
parameter
description | 

text의 x,y 좌표 지정할 위치 |
| s | default 
parameter
description | 

입력할 텍스트 |
| va | default 
parameter
description | center
[ ‘ center’ , ‘ top ‘ , ‘bottom ‘ , ‘baseline’ ]
x,y 좌표 기준 텍스트 위치 지정 |
| ha | default 
parameter
description | center
[ ‘ center’ , ‘ left ‘ , ‘right ‘ ]
x,y 좌표 기준 텍스트 위치 지정 |
| color | default 
parameter
description | None
color
텍스트의 색깔을 입힘. |
| fontsize | default 
parameter
description |  |

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.grid()
ax.tick_params(axis='both',
               labelsize=15)
ax.text(x=0,y=0,
        s="Hello",
        fontsize=30,
        va = 'center', ha = 'left')
plt.show()
```

![- default](jpg/Matplotlib_18.png)

- default

![ax.text(x=0,y=0, s= ‘Hello’, fontsize=30](jpg/Matplotlib_19.png)

ax.text(x=0,y=0, s= ‘Hello’, fontsize=30

- parameter : va(verticallignment)

![- default(top)](jpg/Matplotlib_20.png)

- default(top)

![- center](jpg/Matplotlib_21.png)

- center

![- bottom](jpg/Matplotlib_22.png)

- bottom

![- baseline](jpg/Matplotlib_23.png)

- baseline

- parameter : ha(horizontalalignment)

![- default(center)](jpg/Matplotlib_24.png)

- default(center)

![- left](jpg/Matplotlib_25.png)

- left

![- right](jpg/Matplotlib_26.png)

- right

### ax.set_xticks & ax.set_yticks (labelsize Argument)

![Matplotlib_27](jpg/Matplotlib_27.png)

### ax.set_xticks(Arbitrary Locations)

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
ax.set_xlim([0,10]) # or ax.set_xticks([0,1,5,10]) 두 가지 방법으로 설정 가능!
ax.grid()

plt.show()
```

![- ax.set_xlim( [ 0, 10 ] )](jpg/Matplotlib_28.png)

- ax.set_xlim( [ 0, 10 ] )

![- ax.set_xticks([ 0, 1, 5, 10 ])](jpg/Matplotlib_29.png)

- ax.set_xticks([ 0, 1, 5, 10 ])

![- ax.set_xticks([i for i in range(0, 101, 20)]](jpg/Matplotlib_30.png)

- ax.set_xticks([i for i in range(0, 101, 20)]

- 리스트 컴프리핸션으로도 가능

### ax.set_xticks(Major and Minor Ticks)

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
major_xticks = [ i for i in range(0,101,20)]
minor_xticks = [ i for i in range(0,101, 5)]
ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks,
              minor=True)
ax.grid()

plt.show()
```

![- only major_xticks](jpg/Matplotlib_30.png)

- only major_xticks

![- major_xticks & minor_xticks](jpg/Matplotlib_31.png)

- major_xticks & minor_xticks

### Discrete Colormaps

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
color_list = ['b','g','r','c','m','y']
ax.set_xlim([-1, 1])
ax.set_ylim([-1, len(color_list)])
for c_idx, c in enumerate(color_list):
    ax.text(0, c_idx,
            "color="+c,
            fontsize=20,
            ha='center',color=c)

plt.show()
# tab10 colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
# RGB colors = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
```

![Matplotlib_32](jpg/Matplotlib_32.png)

- Named Colors

![- tab10 colors](Matplotlib_33.png)

- tab10 colors

![- RGB colors](jpg/Matplotlib_34.png)

- RGB colors

### Discrete Colormaps (lut Argument)

```python
import matplotlib.cm as cm

figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
cmap = cm.get_cmap('tab20',lut=20)
for i in range(12):
    ax.scatter(i, i, color=cmap(i), s= 100)
plt.show()
```

![cmap(lut = 20)](jpg/Matplotlib_35.png)

cmap(lut = 20)

- cmap(’rainbow’)

```python
figsize=(7,7)
fig,ax = plt.subplots(figsize=figsize,
                      facecolor='yellowgreen')
n_color = 10
cmap = cm.get_cmap('rainbow',lut=n_color)
ax.set_xlim([-1,1])
ax.set_ylim([-1,n_color])
for c_idx in range(n_color):
    color = cmap(c_idx)
    ax.text(0, c_idx,
            f"color={cmap(c_idx)}",
            fontsize=8,
            ha='center',color=color)
plt.show()
```

![- cmap(’rainbow’)](jpg/Matplotlib_36.png)

- cmap(’rainbow’)

### ax.plot and ax.scatter

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0) # seed 고정을 안할 시 값이 계속 변동됨. 이후 시드는 0으로 고정.

n_data = 100
x_data = np.random.normal(0, 1, (n_data,))
y_data = np.random.normal(0, 1, (n_data,))

fig,ax = plt.subplots(figsize=(7,7),
                      facecolor='yellowgreen')
ax.scatter(x_data,y_data) #  2번째 코드 ax.plot(x_data, y_data, 'o') 
plt.show()
```

![- ax.scatter()](jpg/Matplotlib_37.png)

- ax.scatter()

![- ax.plot() marker 표시 변경 가능(e.g., ‘o’, ‘^’, ‘*’)](jpg/Matplotlib_38.png)

- ax.plot() marker 표시 변경 가능(e.g., ‘o’, ‘^’, ‘*’)

```python
np.random.seed(0)

x_min, x_max = -5, 5
n_data = 300

x_data = np.random.uniform(x_min, x_max, n_data)
y_data = x_data + 0.5*np.random.normal(0, 1, n_data)

pred_x = np.linspace(x_min, x_max, 2)
pred_y = pred_x

fig,ax = plt.subplots(figsize=(10,10),
                      facecolor='yellowgreen')
ax.scatter(x_data, y_data)
ax.plot(pred_x, pred_y,
        color = 'r',
        linewidth=3)
plt.show()
```

![Matplotlib_39](jpg/Matplotlib_39.png)

```python
n_data = 500
x_data = np.random.normal(0, 1, size = (n_data, ))
y_data = np.random.normal(0, 1, size = (n_data, ))
s_arr = np.random.uniform(100, 500, n_data)
c_arr = [np.random.uniform(0, 1, 3) for _ in range(n_data)]

fig,ax = plt.subplots(figsize=(10,10),
                      facecolor='yellowgreen')
ax.scatter(x_data, y_data,
           s=s_arr,
           c=c_arr)
plt.show()
```

![- default](jpg/Matplotlib_40.png)

- default

![ax.scatter(alpha = 0.3)](jpg/Matplotlib_41.png)

ax.scatter(alpha = 0.3)

### Color Array at c Argument

```python
PI = np.pi
n_point = 100
t = np.linspace(-4*PI, 4*PI, n_point)
sin = np.sin(t)

cmap = cm.get_cmap('Reds', lut=n_point)
c_arr = [cmap(c_idx) for c_idx in range(n_point)]
fig,ax = plt.subplots(figsize=(15,10),
                      facecolor='yellowgreen')
ax.scatter(t, sin,
           s=300, c=c_arr)
plt.show()
```

![Matplotlib_42](jpg/Matplotlib_42.png)

### Advanced Markers

```python
n_data = 100
x_data = np.random.normal(0, 1, (n_data, ))
y_data = np.random.normal(0, 1, (n_data, ))

fig,ax = plt.subplots(figsize=(5,5),
                      facecolor='yellowgreen')
ax.scatter(x_data,y_data,
           s=300,
           facecolor='white', # white vs none
           edgecolor='tab:blue',
           linewidth=5)
plt.show()
```

- facecolor vs none
- facecolor에 흰색이 들어감으로써 겹치는 선이 안 보이게 됨.

![Matplotlib_43](jpg/Matplotlib_43.png)

![Matplotlib_44](jpg/Matplotlib_44.png)

### ax.plot(y)

                                                                        

```python
np.random.seed(0)

y_data = np.random.normal(loc=0, scale=1, size=(300,))

fig,ax = plt.subplots(figsize=(10, 5),
                      facecolor = 'yellowgreen')
ax.plot(y_data)

fig.tight_layout(pad=3)
ax.tick_params(labelsize=25)
plt.show()
```

![Matplotlib_45](jpg/Matplotlib_45.png)

### ax.plot(x, y)

```python
np.random.seed(0)

n_data = 100
s_idx = 30
x_data = np.arange(s_idx, s_idx+n_data)
y_data = np.random.normal(0, 1, (n_data, ))

fig,ax = plt.subplots(figsize=(10,5),
                      facecolor='yellowgreen')
ax.plot(x_data,y_data)

fig.tight_layout(pad=3)
x_ticks = np.arange(s_idx, s_idx+n_data+1, 20)
ax.set_xticks(x_ticks)

ax.tick_params(labelsize=25)
ax.grid()

plt.show()
```

![Matplotlib_46](jpg/Matplotlib_46.png)

```python
np.random.seed(0)

x_data = np.random.normal(0, 1, (10, ))
y_data = np.random.normal(0, 1, (10, ))

fig,ax = plt.subplots(figsize=(10, 10),
                      facecolor='yellowgreen')
ax.plot(x_data, y_data)

plt.show()
```

![Matplotlib_47](jpg/Matplotlib_47.png)

### Several Line Plots on One Ax

```python
n_data = 100

random_noise1 = np.random.normal(0, 1, (n_data, )) # 0 : 평균, 1 : 표준편차
random_noise2 = np.random.normal(1, 1, (n_data, ))
random_noise3 = np.random.normal(2, 1, (n_data, ))

fig,ax = plt.subplots(figsize=(10, 7),
                      facecolor='yellowgreen')

ax.plot(random_noise1)
ax.plot(random_noise2)
ax.plot(random_noise3)

ax.tick_params(labelsize=20)

plt.show()
```

![Matplotlib_48](jpg/Matplotlib_48.png)

### Several Line Plots on Different Axes

```python
PI = np.pi
t = np.linspace(-4*PI, 4*PI, 1000).reshape(1, -1)
sin = np.sin(t)
cos = np.cos(t)
tan = np.tan(t)
data = np.vstack((sin, cos, tan))

title_list = [r'%sin(t)$', r'$cos(t)$', r'$tan(t)$']
x_ticks = np.arange(-4*PI, 4*PI+PI, PI)
x_tickslabels = [str(i) + r'$\pi$' for i in range(-4,5)]

fig,axes = plt.subplots(3, 1,
                        figsize = (7, 10),
                        sharex=True, facecolor = 'yellowgreen')

for ax_idx, ax in enumerate(axes.flat):
    ax.plot(t.flatten(), data[ax_idx])
    ax.set_title(title_list[ax_idx],
                 fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid()
    if ax_idx == 2:
        ax.set_ylim([-3, 3])

fig.subplots_adjust(left=0.1, right=0.95,
                    bottom=0.05, top=0.95)
axes[-1].set_xticks(x_ticks)
axes[-1].set_xticklabels(x_tickslabels)

plt.show()
```

![Matplotlib_49](jpg/Matplotlib_49.png)

### ax.axvline and ax.axhline

```python
fig,ax = plt.subplots(figsize=(7,7),
                      facecolor='yellowgreen')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

ax.axvline(x = 1,            # ax.axhline(y = 1)
           color='black',
           linewidth=1)

plt.show()
```

![- axvline(x=1)](jpg/Matplotlib_50.png)

- axvline(x=1)

![- ymax =0.8, ymin = 0.2](jpg/Matplotlib_51.png)

- ymax =0.8, ymin = 0.2

![- axhline(y=1)](jpg/Matplotlib_52.png)

- axhline(y=1)

### Line Styles

```python
x_data = np.array([0, 1])
y_data = x_data

fig,ax = plt.subplots(figsize=(10,10),
                      facecolor='yellowgreen')

ax.plot(x_data, y_data)
ax.plot(x_data, y_data+1,
        linestyle='dotted')
ax.plot(x_data, y_data+2,
        linestyle='dashed')
ax.plot(x_data, y_data+3,
        linestyle='dashdot')

plt.show()
```

![Matplotlib_53](jpg/Matplotlib_53.png)

### Line Styles

```python
PI = np.pi
t = np.linspace(-4*PI, 4*PI, 300)
sin = np.sin(t)

fig,ax = plt.subplots(figsize=(10,7),
                      facecolor='yellowgreen')

ax.plot(t, sin)
ax.axhline(y=1,
           linestyle=':')
ax.axhline(y=-1,
           linestyle=':')

plt.show()
```

![Matplotlib_54](jpg/Matplotlib_54.png)

### Markers

```python
PI = np.pi
t = np.linspace(-4*PI, 4*PI, 100)
sin = np.sin(t)

fig,ax = plt.subplots(figsize=(10,7),
                      facecolor='yellowgreen')

ax.plot(t, sin,
        color='black')
ax.plot(t, sin+1,
        marker='o',
        color='black')
ax.plot(t, sin+2,
        marker='D',
        color='black')
ax.plot(t, sin+3,
        marker='s',
        color='black')

plt.show()
```

![Matplotlib_55](jpg/Matplotlib_55.png)

### Customizing Markers

```python
PI = np.pi
t = np.linspace(-4*PI, 4*PI, 100)
sin = np.sin(t)

fig,ax = plt.subplots(figsize=(10,7),
                      facecolor='yellowgreen')

ax.plot(t, sin+1,
        marker='o',
        color='black',
        markersize=15,
        markerfacecolor='r',
        markeredgecolor='b')

plt.show()
```

![Matplotlib_56](jpg/Matplotlib_56.png)

### fmt Argument

```python
x_data = np.array([1, 2, 3, 4, 5])

fig,ax = plt.subplots(figsize=(10,10),
                      facecolor='yellowgreen')

ax.plot(x_data,
        linestyle=':',
        marker='o',
        color='b')
ax.plot(x_data,':ob')

plt.show()
```

![Matplotlib_57](jpg/Matplotlib_57.png)

### Usage of Legend

| parameter | option | description |
| --- | --- | --- |
| loc | default
parameter
description | ‘best’
best , upper right , upper left, lower left, lower right, right, center right, center left, center right, lower center , upper center, center
legend 부문 위치 선정 |
| bbox_to_anchor | default
parameter
description | 2-tuple : (x location , y location ) , 4-tuple : ( x location, y location, width, height )
2-tuple : (x location , y location )
legend의 위치 좌표로 설정. Legend 위치  |
| ncols | default
parameter
description | 1
int
legend의 열 개수 설정 |
| markerscale | default
parameter
description | 1.0
float
labeling된 marker의 크기 조정 |
| title | default
parameter
description | str or None
str
legend의 제목 설정. |
| title_fontsize | default
parameter
description | None
int or { ‘xx-small’ , ‘x-small’ , ‘small’ , ‘medium’ , ‘large’ , ‘x-large’ , ‘xx-large’ }
legend 제목의 글자 크기 설정. |
| labelspacing | default
parameter
description | 0.5
float
라벨링 간의 간격 조정 |
| columnspacing | default
parameter
description | 2.0
float
열 간의 간격 조정. |
- Basic Usage of Legend

```python
np.random.seed(0)

n_data = 100

random_noise1 = np.random.normal(0, 1, (n_data, ))
random_noise2 = np.random.normal(1, 1, (n_data, ))
random_noise3 = np.random.normal(2, 1, (n_data, ))

fig,ax = plt.subplots(figsize=(10, 7),
                      facecolor='yellowgreen')
ax.tick_params(labelsize=20)

ax.plot(random_noise1,
        label = 'random noise1')
ax.plot(random_noise2,
        label = 'random noise2')
ax.plot(random_noise3,
        label = 'random noise3')
ax.legend() # default loc : best # right : loc = 'upper right'
```

![Matplotlib_58](jpg/Matplotlib_58.png)

![Matplotlib_59](jpg/Matplotlib_59.png)

- ncol Argument

```python
PI = np.pi
t = np.linspace(-4*PI, 4*PI, 100)
sin = np.sin(t)

fig,ax = plt.subplots(figsize=(10,10),
                      facecolor='yellowgreen')

for ax_idx in range(12):
    label_template = 'added by {}'
    ax.plot(t, sin+ax_idx,
            label = label_template.format(ax_idx))
ax.legend(fontsize=15,
          ncol=2)

plt.show()
```

![Matplotlib_60](jpg/Matplotlib_60.png)

![Matplotlib_61](jpg/Matplotlib_61.png)

- bbox_to_anchor Argument

![출처 : 양정은 강사](jpg/Matplotlib_62.png)

출처 : 양정은 강사

```python
np.random.seed(0)

n_data = 100
random_noise1 = np.random.normal(0, 1, (n_data, ))
random_noise2 = np.random.normal(1, 1, (n_data, ))
random_noise3 = np.random.normal(2, 1, (n_data, ))

fig,ax = plt.subplots(figsize=(10, 7),
                      facecolor='yellowgreen')
ax.tick_params(labelsize=20)

ax.plot(random_noise1,
        label = 'random noise1')
ax.plot(random_noise2,
        label = 'random noise2')
ax.plot(random_noise3,
        label = 'random noise3')

ax.legend(fontsize=20,
          bbox_to_anchor = (1, 0.5),
          loc = 'center left')
fig.tight_layout()

plt.show()
```

![Matplotlib_63](jpg/Matplotlib_63.png)

### Advanced Legend

```python
np.random.seed(0)

n_class = 5
n_data = 30
center_pt = np.random.uniform(-20, 20, (n_class, 2))
cmap = cm.get_cmap('tab20')
colors = [cmap(i) for i in range(n_class)]

data_dict = {'class'+str(i) : None for i in range(n_class)}
for class_idx in range(n_class):
    center = center_pt[class_idx]

    x_data = center[0] + 2 * np.random.normal(0, 1, (1, n_data))
    y_data = center[1] + 2 * np.random.normal(0, 1, (1, n_data))
    data = np.vstack((x_data, y_data))

    data_dict['class' + str(class_idx)] = data

fig, ax = plt.subplots(figsize=(12, 10),
                       facecolor='yellowgreen')
for class_idx in range(n_class):
    data = data_dict['class' + str(class_idx)]
    ax.scatter(data[0], data[1],
               s=1000,
               facecolor='None',
               edgecolor=colors[class_idx],
               linewidth=5,
               alpha=0.5,
               label='class' + str(class_idx))

ax.legend(loc='center left',
          bbox_to_anchor = (1, 0.5),
          fontsize = 10,
          markerscale=0.5,
          ncol=2)

fig.tight_layout()
plt.show()
```

![Matplotlib_64](jpg/Matplotlib_64.png)

### Practice

- 1번.

![Matplotlib_65](jpg/Matplotlib_65.png)

```python
countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile',
             'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
             'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland',
             'Israel', 'Italy', 'Japan', 'Korea', 'Luxembourg',
             'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland',
             'Portuagl', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden',
             'Switzerland', 'Turkey', 'United Kingdom', 'United States']

population_density = [3, 101, 367, 4, 23,
                      133, 130, 29, 16, 117,
                      227, 86, 106, 3, 65,
                      365, 203, 337, 501, 207,
                      60, 406, 17, 13, 122,
                      116, 110, 103, 91, 21,
                      194, 97, 257, 32]
private_expenditure = [1.3, 8.9, 10.0, 10.9, 12.8,
                       13.0, 13.2, 13.4, 13.5, 14.8,
                       15.0, 15.8, 16.7, 17.1, 17.2,
                       17.4, 18.1, 18.1, 18.7, 18.9,
                       19.0, 19.0, 19.5, 19.9, 20.0,
                       21.5, 21.6, 23.0, 24.0, 25.0,
                       25.9, 26.7, 28.0, 34.3]
gdp = [38.7, 37.4, 33.6, 37.5, 16.4,
       24.5, 33.2, 19.3, 32.1, 32.0,
       36.2, 19.8, 17.8, 37.7, 37.7,
       29.4, 26.6, 32.0, 31.0, 67.9,
       13.4, 38.4, 27.0, 48.2, 18.9,
       20.9, 21.8, 24.2, 26.8, 36.2,
       42.5, 13.9, 35.6, 45.7]

lst = []
lst_legend = np.array([10,25,40,55])
gdp = np.array(gdp).reshape(2,-1)
population_denstiy = np.array(population_denstiy).reshape(2,-1)
private_expenditure = np.array(private_expenditure).reshape(2,-1)
countries = np.array(countries).reshape(2,-1)
lst_hatch = ['//','.']
lst_ax = []
n = len(countries)
n_half = len(countries) // 2

cmap = cm.get_cmap('tab20',lut=17)

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot()
ax1 = ax.twinx()

plt.gca().axes.yaxis.set_visible(False)
ax.set_xlabel('Population Density(Inh./km2)',
              fontsize=20)
ax.set_ylabel('Private Expenditure',
              fontsize=20)
for i in range(2):
    for j in range(17):
        ax.scatter(population_denstiy[i][j],
                   private_expenditure[i][j],
                   color=cmap(j),
                   s = gdp[i][j]*25,
                   hatch=lst_hatch[i],
                   edgecolor='None',
                   alpha = 0.8)
        ax.scatter([],[],
                   color=cmap(j),
                   label=countries[i][j],
                   s = 500,
                   hatch=lst_hatch[i],
                   alpha=0.8,
                   edgecolor='None')
for i in range(4):
    ax1.scatter([],[],
                color = cmap(3),
                s = lst_legend[i]*50,
                label = lst_legend[i])
ax.grid()
ax.legend(loc='center left',
          fontsize=10,
          ncol=2,
          bbox_to_anchor=(1,0.5),
          markerscale=0.5,
          labelspacing=1)
ax1.legend(title='GDP Value',
           title_fontsize=30,
           loc='lower center',
           fontsize=10,
           markerscale=0.5,
           bbox_to_anchor=(0.5,1),
           labelspacing=2,
           columnspacing=1,
           edgecolor='white',
           ncol=4)
fig.tight_layout()
plt.show()
```

![- result](jpg/Matplotlib_66.png)

- result

- 2번.

![Matplotlib_67](jpg/Matplotlib_67.png)

```python
np.random.seed(8)
cmap = cm.get_cmap('tab20',lut=20)
n_data = 200
x_min, x_max = -60,60

fig,ax = plt.subplots(figsize=(12,12))

raw_data_1 = np.random.uniform(x_min,x_max,n_data)
raw_data_2 = np.random.uniform(x_min,x_max,n_data)
for idx in range(20):
    data1 = np.random.uniform(raw_data_1[idx]-1,raw_data_1[idx]+1,n_data)
    data2 = np.random.uniform(raw_data_2[idx]-1,raw_data_2[idx]+1,n_data)
    x_data = data1 + 0.5 *np.random.normal(0, 3, (n_data))
    y_data = data2 + 0.5 *np.random.normal(0, 3, (n_data))
    for data_idx in range(200):
        ax.scatter(x_data[data_idx],y_data[data_idx],
                   s = 50,
                   color = cmap(idx))
plt.show()

fig = plt.figure(figsize=(7,7),
                 facecolor='yellowgreen')
ax = fig.add_subplot()
plt.show()
```

![Matplotlib_68](jpg/Matplotlib_68.png)

- 3번.

![Matplotlib_69](jpg/Matplotlib_69.png)

```python
names = ['DFF R-FCN', 'R-FCN', 'FGFA R-FCN'] # label에 들어갈 내용
dff_data = np.array([(0.581, 13.5),(0.598, 12.8),(0.618, 11.7),
           (0.62, 11.3), (0.624, 10.2), (0.627, 9.8),
           (0.629, 9.2), (0.63, 9)])
r_data = np.array([(0.565, 11.2), (0.645, 9)])
fgfa_data = np.array([(0.63, 8.8), (0.653, 9.3), (0.664, 9.6),
           (0.676, 10.1)])
dff_text = ['1:20', '1:15', '1:10', '1:8',
            '1:5', '1:3', '1:2', '1:1']
r_text = ['Half Model', 'Full Model']
fgfa_text = ['1:1', '3:1', '7:1', '19:1']
lst_label = ['o','^','*']

cmap = cm.get_cmap('tab20',lut=3)
fig,ax = plt.subplots(figsize=(15,10))
ax.set_xlabel('mAP',
              fontsize=20)
ax.set_ylabel('AD',
              fontsize=20)
for idx in range(len(dff_data)):
    ax.plot(dff_data[idx][0],dff_data[idx][1],
            'o',
            color=cmap(0),
            markersize=20)
    ax.text(dff_data[idx][0]+0.002,dff_data[idx][1]+0.01,
            dff_text[idx],
            fontsize=20,
            ha='left',va='bottom')
for idx in range(len(r_text)):
    ax.plot(r_data[idx][0],r_data[idx][1],
            '^',
            markersize=25,
            color=cmap(1))
    ax.text(r_data[idx][0]-0.004,r_data[idx][1]-0.15,
            r_text[idx],
            fontsize = 15,
            va='top')
for fgfa_idx in range(len(fgfa_data)):
    ax.plot(fgfa_data[fgfa_idx][0],fgfa_data[fgfa_idx][1],
            '*',
            markersize = 30,
            color=cmap(2))

    ax.text(fgfa_data[fgfa_idx][0]-0.0015,fgfa_data[fgfa_idx][1]-0.1,
            va ='top', ha= 'left',
            fontsize = 15,
            s = fgfa_text[fgfa_idx])
ax.grid()

for idx in range(len(names)):
    ax.plot([],[],
            lst_label[idx],
            label = names[idx],
            color = cmap(idx))

ax.legend(loc='upper right',
          markerscale=3,
          fontsize=30)
fig.tight_layout()
plt.show()
```

![Matplotlib_70](jpg/Matplotlib_70.png)
