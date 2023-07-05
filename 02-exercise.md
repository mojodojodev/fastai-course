# 02: Production
Can see app live [here on hugging face](https://huggingface.co/spaces/mojodojodev/football-classifier) and here's the iframe:

<iframe
	src="https://mojodojodev-football-classifier.hf.space"
	frameborder="0"
	width="850"
	height="550"
></iframe>


This will be improving the previous model with new techniques and putting it into production with an API route to call at the end from this page.

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

Use the same dataset from previous exercise


```python
data_path = Path('data')/'01'
data_path.mkdir(parents=True,exist_ok=True)
```

## Download Dataset


```python
searches = "rugby","afl","nfl","soccer"
dataset_path = data_path/"datasets"
```

## Build Data Block


```python
failed = verify_images(get_image_files(dataset_path))
failed.map(Path.unlink)
```




    (#0) []




```python
football = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
)

dls = football.dataloaders(dataset_path)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/02-exercise_10_0.png)
    


## Augment Images


```python
football = football.new(
    item_tfms=RandomResizedCrop(192, min_scale=0.5),
    batch_tfms=aug_transforms()
)
dls = football.dataloaders(dataset_path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```


    
![png](/02-exercise_12_0.png)
    


## Train Model


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(8)
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
      <td>1.892716</td>
      <td>1.607285</td>
      <td>0.551724</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.783888</td>
      <td>1.293517</td>
      <td>0.448276</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.553403</td>
      <td>1.101768</td>
      <td>0.396552</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.386793</td>
      <td>1.080031</td>
      <td>0.344828</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.214310</td>
      <td>1.075227</td>
      <td>0.327586</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.076431</td>
      <td>1.076623</td>
      <td>0.310345</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.965085</td>
      <td>1.075264</td>
      <td>0.275862</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.882993</td>
      <td>1.080741</td>
      <td>0.258621</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## Confusion Matrix


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```






    
![png](/02-exercise_16_1.png)
    


## Show examples of what's causing problems


```python
interp.plot_top_losses(5, nrows=1, figsize=(17,4))
```






    
![png](/02-exercise_18_1.png)
    



```python
sport,_,probs = learn.predict(PILImage.create(data_path/'nfl.jpg'))
print(f"this is {sport}")
print("\n--probabilities--")
for i, o in enumerate(dls.vocab):
    print(f"{o}: {probs[i]:.4f}")
```





    this is nfl
    
    --probabilities--
    afl: 0.0000
    nfl: 1.0000
    rugby: 0.0000
    soccer: 0.0000


## Export the model


```python
learn.export("football-classifier/model.pkl")
learn.export("model.pkl")
```

## Build script to put it into production on HuggingFace


```python
#|default_exp app
```


```python
#|export
from fastai.vision.all import *
import gradio as gr
```


```python
#|export
learn = load_learner('model.pkl')
```


```python
#|export
categories = ('afl', 'nfl', 'rugby', 'soccer')

def classify_image(img):
    _,_,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))
```


```python
im = PILImage.create('data/01/afl.jpg')
im.thumbnail((192,192))
im
```




    
![png](/02-exercise_27_0.png)
    




```python
classify_image(im)
```








    {'afl': 0.9970560073852539,
     'nfl': 0.0002015349455177784,
     'rugby': 0.0019157134229317307,
     'soccer': 0.0008266967488452792}




```python
#|export
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = [f'data/01/{i}.jpg' for i in categories]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
```

    /home/j/src/fastai-course/.venv/lib/python3.11/site-packages/gradio/inputs.py:259: UserWarning: Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components
      warnings.warn(
    /home/j/src/fastai-course/.venv/lib/python3.11/site-packages/gradio/inputs.py:262: UserWarning: `optional` parameter is deprecated, and it has no effect
      super().__init__(
    /home/j/src/fastai-course/.venv/lib/python3.11/site-packages/gradio/outputs.py:197: UserWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components
      warnings.warn(
    /home/j/src/fastai-course/.venv/lib/python3.11/site-packages/gradio/outputs.py:200: UserWarning: The 'type' parameter has been deprecated. Use the Number component instead.
      super().__init__(num_top_classes=num_top_classes, type=type, label=label)


    Running on local URL:  http://127.0.0.1:7861
    
    To create a public link, set `share=True` in `launch()`.





    



## Export the script ready to push to Hugging Face


```python
from nbdev import export

export.nb_export('02-exercise.ipynb', 'football-classifier')
```
