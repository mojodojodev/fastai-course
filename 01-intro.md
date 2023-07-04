# Intro
Raw code for experimentation


```python
from fastbook import *
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
```

## Classifier


```python
download_url(search_images_ddg('bird photos', max_images=1)[0], 'bird.jpg')
Image.open('bird.jpg').to_thumb(256, 256)
```



<div>
  <progress value='376832' class='' max='374309' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.67% [376832/374309 00:00&lt;00:00]
</div>






    
![png](/01-intro_3_1.png)
    




```python
download_url(search_images_ddg('forest photos', max_images=1)[0], 'forest.jpg')
Image.open('forest.jpg').to_thumb(256, 256)
```



<div>
  <progress value='532480' class='' max='529125' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.63% [532480/529125 00:01&lt;00:00]
</div>






    
![png](/01-intro_4_1.png)
    




```python
searches = 'forest','bird'
path = Path('bird_or_not')
```


```python
if not path.exists():
    for o in searches:
        dest = (path/o)
        dest.mkdir(parents=True,exist_ok=True)
        results = search_images_ddg(f'{o} photo')
        download_images(dest, urls=results[:200])
        resize_images(dest, max_size=400, dest=dest)
```


```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
```




    (#0) []




```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)
```


    
![png](/01-intro_8_0.png)
    



```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
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
      <td>0.055454</td>
      <td>0.035612</td>
      <td>0.013158</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.031750</td>
      <td>0.001773</td>
      <td>0.000000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.020932</td>
      <td>0.001520</td>
      <td>0.000000</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
is_bird,_,probs = learn.predict(PILImage.create('bird_or_not/forest/forest.jpg'))
print(f"This is a: {is_bird}")
print(f"probability it's a bird: {probs[0]:.4f}")
print(f"probability it's a forest: {probs[1]:.4f}")
```





    This is a: forest
    probability it's a bird: 0.0000
    probability it's a forest: 1.0000


## Segmentation


```python
import torch
torch.cuda.empty_cache()
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(120)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[10], line 11
          4 dls = SegmentationDataLoaders.from_label_func(
          5     path, bs=8, fnames = get_image_files(path/"images"),
          6     label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
          7     codes = np.loadtxt(path/'codes.txt', dtype=str)
          8 )
         10 learn = unet_learner(dls, resnet34)
    ---> 11 learn.fine_tune(120)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/callback/schedule.py:168, in fine_tune(self, epochs, base_lr, freeze_epochs, lr_mult, pct_start, div, **kwargs)
        166 base_lr /= 2
        167 self.unfreeze()
    --> 168 self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/callback/schedule.py:119, in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
        116 lr_max = np.array([h['lr'] for h in self.opt.hypers])
        117 scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
        118           'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    --> 119 self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:264, in Learner.fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
        262 self.opt.set_hypers(lr=self.lr if lr is None else lr)
        263 self.n_epoch = n_epoch
    --> 264 self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:199, in Learner._with_events(self, f, event_type, ex, final)
        198 def _with_events(self, f, event_type, ex, final=noop):
    --> 199     try: self(f'before_{event_type}');  f()
        200     except ex: self(f'after_cancel_{event_type}')
        201     self(f'after_{event_type}');  final()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:253, in Learner._do_fit(self)
        251 for epoch in range(self.n_epoch):
        252     self.epoch=epoch
    --> 253     self._with_events(self._do_epoch, 'epoch', CancelEpochException)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:199, in Learner._with_events(self, f, event_type, ex, final)
        198 def _with_events(self, f, event_type, ex, final=noop):
    --> 199     try: self(f'before_{event_type}');  f()
        200     except ex: self(f'after_cancel_{event_type}')
        201     self(f'after_{event_type}');  final()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:247, in Learner._do_epoch(self)
        246 def _do_epoch(self):
    --> 247     self._do_epoch_train()
        248     self._do_epoch_validate()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:239, in Learner._do_epoch_train(self)
        237 def _do_epoch_train(self):
        238     self.dl = self.dls.train
    --> 239     self._with_events(self.all_batches, 'train', CancelTrainException)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:199, in Learner._with_events(self, f, event_type, ex, final)
        198 def _with_events(self, f, event_type, ex, final=noop):
    --> 199     try: self(f'before_{event_type}');  f()
        200     except ex: self(f'after_cancel_{event_type}')
        201     self(f'after_{event_type}');  final()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:205, in Learner.all_batches(self)
        203 def all_batches(self):
        204     self.n_iter = len(self.dl)
    --> 205     for o in enumerate(self.dl): self.one_batch(*o)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:235, in Learner.one_batch(self, i, b)
        233 b = self._set_device(b)
        234 self._split(b)
    --> 235 self._with_events(self._do_one_batch, 'batch', CancelBatchException)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:201, in Learner._with_events(self, f, event_type, ex, final)
        199 try: self(f'before_{event_type}');  f()
        200 except ex: self(f'after_cancel_{event_type}')
    --> 201 self(f'after_{event_type}');  final()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:172, in Learner.__call__(self, event_name)
    --> 172 def __call__(self, event_name): L(event_name).map(self._call_one)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastcore/foundation.py:156, in L.map(self, f, *args, **kwargs)
    --> 156 def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastcore/basics.py:840, in map_ex(iterable, f, gen, *args, **kwargs)
        838 res = map(g, iterable)
        839 if gen: return res
    --> 840 return list(res)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastcore/basics.py:825, in bind.__call__(self, *args, **kwargs)
        823     if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
        824 fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
    --> 825 return self.func(*fargs, **kwargs)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/learner.py:176, in Learner._call_one(self, event_name)
        174 def _call_one(self, event_name):
        175     if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
    --> 176     for cb in self.cbs.sorted('order'): cb(event_name)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/callback/core.py:60, in Callback.__call__(self, event_name)
         58 res = None
         59 if self.run and _run: 
    ---> 60     try: res = getcallable(self, event_name)()
         61     except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
         62     except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/callback/progress.py:32, in ProgressCallback.after_batch(self)
         31 def after_batch(self):
    ---> 32     self.pbar.update(self.iter+1)
         33     if hasattr(self, 'smooth_loss'): self.pbar.comment = f'{self.smooth_loss.item():.4f}'


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastprogress/fastprogress.py:66, in ProgressBar.update(self, val)
         64 self.pred_t = None if self.total is None else avg_t * self.total
         65 self.last_v,self.last_t = val,cur_t
    ---> 66 self.update_bar(val)
         67 if self.total is not None and val >= self.total:
         68     self.on_iter_end()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastprogress/fastprogress.py:81, in ProgressBar.update_bar(self, val)
         79 elapsed_t = format_time(elapsed_t)
         80 end = '' if len(self.comment) == 0 else f' {self.comment}'
    ---> 81 self.on_update(val, f'{pct}[{val}/{tot} {elapsed_t}{self.lt}{remaining_t}{end}]')


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastprogress/fastprogress.py:134, in NBProgressBar.on_update(self, val, text, interrupted)
        132 self.progress = html_progress_bar(val, self.total, text, interrupted)
        133 if self.display: self.out.update(HTML(self.progress))
    --> 134 elif self.parent is not None: self.parent.show()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastprogress/fastprogress.py:177, in NBMasterBar.show(self)
        175 to_show = [name for name in self.order if name in self.inner_dict.keys()]
        176 self.html_code = '\n'.join([getattr(self.inner_dict[n], 'progress', self.inner_dict[n]) for n in to_show])
    --> 177 self.out.update(HTML(self.html_code))


    Cell In[1], line 5, in update_patch(self, obj)
          3 def update_patch(self, obj):
          4     clear_output(wait=True)
    ----> 5     self.display(obj)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/IPython/core/display_functions.py:362, in DisplayHandle.display(self, obj, **kwargs)
        352 def display(self, obj, **kwargs):
        353     """Make a new display with my id, updating existing instances.
        354 
        355     Parameters
       (...)
        360         additional keyword arguments passed to display
        361     """
    --> 362     display(obj, display_id=self.display_id, **kwargs)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/IPython/core/display_functions.py:305, in display(include, exclude, metadata, transient, display_id, raw, clear, *objs, **kwargs)
        302         if metadata:
        303             # kwarg-specified metadata gets precedence
        304             _merge(md_dict, metadata)
    --> 305         publish_display_data(data=format_dict, metadata=md_dict, **kwargs)
        306 if display_id:
        307     return DisplayHandle(display_id)


    File ~/src/fastbook/venv/lib/python3.11/site-packages/IPython/core/display_functions.py:93, in publish_display_data(data, metadata, source, transient, **kwargs)
         90 if transient:
         91     kwargs['transient'] = transient
    ---> 93 display_pub.publish(
         94     data=data,
         95     metadata=metadata,
         96     **kwargs
         97 )


    File ~/src/fastbook/venv/lib/python3.11/site-packages/ipykernel/zmqshell.py:102, in ZMQDisplayPublisher.publish(self, data, metadata, transient, update)
         80 def publish(
         81     self,
         82     data,
       (...)
         85     update=False,
         86 ):
         87     """Publish a display-data message
         88 
         89     Parameters
       (...)
        100         If True, send an update_display_data message instead of display_data.
        101     """
    --> 102     self._flush_streams()
        103     if metadata is None:
        104         metadata = {}


    File ~/src/fastbook/venv/lib/python3.11/site-packages/ipykernel/zmqshell.py:66, in ZMQDisplayPublisher._flush_streams(self)
         64 """flush IO Streams prior to display"""
         65 sys.stdout.flush()
    ---> 66 sys.stderr.flush()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/ipykernel/iostream.py:497, in OutStream.flush(self)
        495     self.pub_thread.schedule(evt.set)
        496     # and give a timeout to avoid
    --> 497     if not evt.wait(self.flush_timeout):
        498         # write directly to __stderr__ instead of warning because
        499         # if this is happening sys.stderr may be the problem.
        500         print("IOStream.flush timed out", file=sys.__stderr__)
        501 else:


    File ~/miniforge3/envs/vscode/lib/python3.11/threading.py:622, in Event.wait(self, timeout)
        620 signaled = self._flag
        621 if not signaled:
    --> 622     signaled = self._cond.wait(timeout)
        623 return signaled


    File ~/miniforge3/envs/vscode/lib/python3.11/threading.py:324, in Condition.wait(self, timeout)
        322 else:
        323     if timeout > 0:
    --> 324         gotit = waiter.acquire(True, timeout)
        325     else:
        326         gotit = waiter.acquire(False)


    KeyboardInterrupt: 



```python
learn.show_results(max_n=3, figsize=(7,8))
```






    
![png](/01-intro_13_1.png)
    


## Tabular Analysis


```python
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names='salary',
                                  cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship'],
                                  cont_names = ['age', 'fnlwgt', 'education-num'],
                                  procs = [Categorify, FillMissing, Normalize]
                                  )
```


```python
dls.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>False</td>
      <td>25.0</td>
      <td>262664.001254</td>
      <td>13.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>False</td>
      <td>43.0</td>
      <td>55394.998497</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Self-emp-not-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>False</td>
      <td>51.0</td>
      <td>156801.999536</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Federal-gov</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>False</td>
      <td>45.0</td>
      <td>181969.999731</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>False</td>
      <td>38.0</td>
      <td>411797.005132</td>
      <td>11.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Unmarried</td>
      <td>False</td>
      <td>39.0</td>
      <td>158955.999220</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Local-gov</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>False</td>
      <td>50.0</td>
      <td>231724.999023</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Unmarried</td>
      <td>False</td>
      <td>43.0</td>
      <td>115806.000281</td>
      <td>14.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>11th</td>
      <td>Divorced</td>
      <td>Craft-repair</td>
      <td>Not-in-family</td>
      <td>False</td>
      <td>56.0</td>
      <td>271794.998970</td>
      <td>7.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Tech-support</td>
      <td>Not-in-family</td>
      <td>False</td>
      <td>46.0</td>
      <td>295791.002851</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>



```python
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(2)
learn.show_results()
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In[19], line 1
    ----> 1 learn = tabular_learner(dls, metrics=accuracy)
          2 learn.fit_one_cycle(2)
          3 learn.show_results()


    File ~/src/fastbook/venv/lib/python3.11/site-packages/fastai/tabular/learner.py:44, in tabular_learner(dls, layers, emb_szs, config, n_out, y_range, **kwargs)
         42 emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
         43 if n_out is None: n_out = get_c(dls)
    ---> 44 assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
         45 if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
         46 model = TabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, **config)


    AssertionError: `n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`


## Recommendations


```python
from fastai.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
```


```python
dls.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>130</td>
      <td>293</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>505</td>
      <td>480</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>402</td>
      <td>5952</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>239</td>
      <td>1073</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>380</td>
      <td>344</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>306</td>
      <td>733</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>468</td>
      <td>364</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>665</td>
      <td>4306</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17</td>
      <td>541</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>615</td>
      <td>6377</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>



```python
learn = collab_learner(dls, y_range=(0.5, 5.5))
learn.fine_tune(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.374860</td>
      <td>1.350394</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.287999</td>
      <td>1.169781</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.031497</td>
      <td>0.867560</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.796313</td>
      <td>0.731387</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.688329</td>
      <td>0.698444</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.642904</td>
      <td>0.692010</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.622824</td>
      <td>0.689128</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.612655</td>
      <td>0.687250</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.605519</td>
      <td>0.686323</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.595020</td>
      <td>0.686292</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
learn.show_results()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>rating_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.0</td>
      <td>74.0</td>
      <td>5.0</td>
      <td>3.763442</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.0</td>
      <td>75.0</td>
      <td>4.0</td>
      <td>4.305429</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.0</td>
      <td>9.0</td>
      <td>1.5</td>
      <td>3.140058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>98.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>3.774874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.0</td>
      <td>94.0</td>
      <td>4.0</td>
      <td>3.496572</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>57.0</td>
      <td>4.0</td>
      <td>3.656664</td>
    </tr>
    <tr>
      <th>6</th>
      <td>53.0</td>
      <td>18.0</td>
      <td>5.0</td>
      <td>4.607825</td>
    </tr>
    <tr>
      <th>7</th>
      <td>79.0</td>
      <td>45.0</td>
      <td>5.0</td>
      <td>4.669981</td>
    </tr>
    <tr>
      <th>8</th>
      <td>91.0</td>
      <td>71.0</td>
      <td>4.0</td>
      <td>3.118788</td>
    </tr>
  </tbody>
</table>

