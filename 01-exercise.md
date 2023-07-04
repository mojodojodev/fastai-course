# 01: Football Code Classifier
## Setup
Import fastbook and patch IPython for VS Code


```python
from fastbook import *
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
```

Set up a path to store the images


```python
data_path = Path('data')/'01'
data_path.mkdir(parents=True,exist_ok=True)
```

## Preview Images
Set up a function to try out different keywords and see how they look


```python
def preview(keyword: string):
    path = data_path / f"{keyword}.jpg"
    download_url(search_images_ddg(keyword, max_images=1)[0], path)
    return Image.open(path).to_thumb(256, 256)
```


```python
preview("rugby")
```



<div>
  <progress value='98304' class='' max='97571' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.75% [98304/97571 00:02&lt;00:00]
</div>






    
![png](/01-exercise_6_1.png)
    




```python
preview("nfl")
```



<div>
  <progress value='245760' class='' max='243610' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.88% [245760/243610 00:00&lt;00:00]
</div>






    
![png](/01-exercise_7_1.png)
    




```python
preview("soccer")
```



<div>
  <progress value='81920' class='' max='75868' style='width:300px; height:20px; vertical-align: middle;'></progress>
  107.98% [81920/75868 00:00&lt;00:00]
</div>






    
![png](/01-exercise_8_1.png)
    




```python
preview("afl")
```



<div>
  <progress value='139264' class='' max='139007' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.18% [139264/139007 00:00&lt;00:00]
</div>






    
![png](/01-exercise_9_1.png)
    



## Download Dataset


```python
searches = "rugby","afl","nfl","soccer"
dataset_path = data_path/"datasets"
```


```python
if not (dataset_path).exists():
    for o in searches:
        dest = (dataset_path/o)
        dest.mkdir(parents=True,exist_ok=True)
        results = search_images_ddg(f"{o} game")
        download_images(dest, urls=results[:200])
        resize_images(dest, max_size=400, dest=dest)
```

## Build Data Block


```python
failed = verify_images(get_image_files(dataset_path))
failed.map(Path.unlink)
```




    (#17) [None,None,None,None,None,None,None,None,None,None...]




```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(dataset_path)

dls.show_batch(max_n=6)
```


    
![png](/01-exercise_15_0.png)
    


## Train Model


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(30)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.331276</td>
      <td>1.282537</td>
      <td>0.398230</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.073042</td>
      <td>1.108479</td>
      <td>0.362832</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.873794</td>
      <td>0.998766</td>
      <td>0.371681</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.718180</td>
      <td>0.987133</td>
      <td>0.389381</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.583215</td>
      <td>1.016907</td>
      <td>0.362832</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.476426</td>
      <td>1.067516</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.383498</td>
      <td>1.137339</td>
      <td>0.362832</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.314319</td>
      <td>1.200328</td>
      <td>0.353982</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.261772</td>
      <td>1.208668</td>
      <td>0.336283</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.218976</td>
      <td>1.175529</td>
      <td>0.327434</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.184930</td>
      <td>1.139289</td>
      <td>0.300885</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.156622</td>
      <td>1.109112</td>
      <td>0.292035</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.133540</td>
      <td>1.112555</td>
      <td>0.292035</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.114173</td>
      <td>1.126750</td>
      <td>0.300885</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.098229</td>
      <td>1.136930</td>
      <td>0.300885</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.084903</td>
      <td>1.140284</td>
      <td>0.318584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.074624</td>
      <td>1.148302</td>
      <td>0.327434</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.064603</td>
      <td>1.167305</td>
      <td>0.318584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.055911</td>
      <td>1.179664</td>
      <td>0.318584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.048461</td>
      <td>1.191116</td>
      <td>0.327434</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.042165</td>
      <td>1.178158</td>
      <td>0.336283</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.036902</td>
      <td>1.185483</td>
      <td>0.327434</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.032348</td>
      <td>1.170821</td>
      <td>0.336283</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.028613</td>
      <td>1.168687</td>
      <td>0.336283</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.025399</td>
      <td>1.167978</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.022398</td>
      <td>1.166666</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.019721</td>
      <td>1.173368</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.017580</td>
      <td>1.174313</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.015551</td>
      <td>1.172876</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.013663</td>
      <td>1.171674</td>
      <td>0.345133</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## Test Model


```python
learn.show_results()
```






    
![png](/01-exercise_19_1.png)
    



```python
sport,_,probs = learn.predict(PILImage.create(data_path/'nfl.jpg'))
print(f"this is {sport}")
print("\n--probabilities--")
for i, o in enumerate(dls.vocab):
    print(f"{o}: {probs[i]:.4f}")
```





    this is nfl
    
    --probabilities--
    afl: 0.0008
    nfl: 0.9891
    rugby: 0.0013
    soccer: 0.0089

