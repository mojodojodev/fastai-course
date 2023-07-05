# 02: Production
## Setup
Import fastbook and patch IPython for VS Code


```python
from fastbook import *
from fastai.vision.widgets import *
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
```

Set up a path to store the images


```python
data_path = Path('data')/'02'
data_path.mkdir(parents=True,exist_ok=True)
```

## Download Images


```python
results = search_images_ddg('grizzly bear')
ims=results[:200]
len(ims)
```




    200




```python
dest = data_path/'grizzly.jpg'
download_url(ims[0], dest)
```



<div>
  <progress value='1220608' class='' max='1218659' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.16% [1220608/1218659 00:03&lt;00:00]
</div>






    Path('data/02/grizzly.jpg')




```python
im = Image.open(dest)
im.to_thumb(128, 128)
```




    
![png](/02-production_7_0.png)
    




```python
bear_types = 'grizzly','black','teddy'
path = data_path/'bears'
```


```python
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f'{o} bear')
        download_images(dest, urls=results[:200])
```


```python
fns = get_image_files(path)
```


```python
failed = verify_images(fns)
failed.map(Path.unlink)
```




    (#0) []




```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2,seed=42),
    get_y=parent_label,
    item_tfms=Resize(128),
)
```


```python
dls = bears.dataloaders(path)
```


```python
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/02-production_14_0.png)
    



```python
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/02-production_15_0.png)
    



```python
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```


    
![png](/02-production_16_0.png)
    



```python
bear = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```


    
![png](/02-production_17_0.png)
    



```python
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms()
)
dls = bears.dataloaders(path)
```


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
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
      <td>0.387830</td>
      <td>0.063339</td>
      <td>0.027778</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.318884</td>
      <td>0.043817</td>
      <td>0.018519</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.276268</td>
      <td>0.042250</td>
      <td>0.018519</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.237925</td>
      <td>0.047473</td>
      <td>0.018519</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>



```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```






    
![png](/02-production_20_1.png)
    



```python
interp.plot_top_losses(5, nrows=1, figsize=(17,4))
```






    
![png](/02-production_21_1.png)
    



```python
cleaner = ImageClassifierCleaner(learn)
cleaner
```






    VBox(children=(Dropdown(options=('black', 'grizzly', 'teddy'), value='black'), Dropdown(options=('Train', 'Val…



```python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```


```python
learn.export("model.pkl")
```
