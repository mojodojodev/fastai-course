## enable progress bars in VSCode
```python
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
```

## How to fetch from gradio api
<script setup>
    import { client } from "@gradio/client";
    const run = async () => {
        console.log("getting data")
        const response_0 = await fetch("https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png");
        const exampleImage = await response_0.blob();
                            
        const app = await client("https://mojodojodev-minimal.hf.space/");
        const result = await app.predict("/predict", [
                    exampleImage, 	// blob in 'img' Image component
        ]);

        console.log(result?.data);
    }
</script>


  <div>
    <button @click="run">Run Prediction</button>
  </div>
