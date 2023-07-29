## enable progress bars in VSCode
```python
from IPython.display import clear_output, DisplayHandle
def update_patch(self, obj):
    clear_output(wait=True)
    self.display(obj)
DisplayHandle.update = update_patch
```

## How to fetch from gradio api
```html
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
  ```

## Suggestions to understand backpropogation from GPT4

I completely understand your curiosity! Deep learning and its implementations can indeed be quite fascinating. If you're looking to understand the internals of how backpropagation works, and more broadly, how deep learning models learn, here are some resources I'd recommend:

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville: This is often referred to as the "bible" of deep learning. It covers everything from the basics to the very latest research.
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron: This book offers a practical introduction to machine learning and deep learning, and includes many examples and exercises.

### Online Courses
- Andrew Ng's Deep Learning Specialization on Coursera: This is a comprehensive introduction to deep learning spread across five courses. Course 1 and 2 cover the basics of neural networks and deep learning, and they include detailed explanations of backpropagation.
- Fast.ai courses: Fast.ai offers two free courses, "Practical Deep Learning for Coders" and "Deep Learning from the Foundations". The latter one goes into the details of how deep learning libraries work under the hood.

### Blogs and Websites
- Chris Olah's Blog: Chris Olah has written some great articles explaining the inner workings of neural networks. His post on understanding LSTM Networks is a classic.
- The Backpropagation Algorithm Demystified: This is a Medium article by Niklas Donges that breaks down the mathematics of backpropagation.
- Neural networks and deep learning by Michael Nielsen: This is an online book that's very readable and does a good job explaining the basics, including backpropagation.

### Research Papers
- "Learning Representations by Back-propagating Errors" by Rumelhart, Hinton, and Williams: This is the original paper that introduced the backpropagation algorithm.

Remember that understanding backpropagation and the inner workings of deep learning libraries requires a good understanding of calculus, especially the chain rule, as well as linear algebra. So if you're not already comfortable with these areas, you may want to brush up on them. Khan Academy provides excellent courses in both these areas.
