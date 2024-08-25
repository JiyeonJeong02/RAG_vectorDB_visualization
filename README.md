# RAG_vectorDB_visualization
VectorDB visualization for RAG project (using TensorBoard and MNIST Dataset)

---

The dataset used in this project is the "MNIST Dataset," which contains images of handwritten digits (0-9).
The details of the dataset are as follows:
- 28x28 grayscale images → both width and height are 28 pixels.
- Training data: 60,000 samples
- Test data: 10,000 samples

As you can see in the code, it was taken from [Anuj Shah](https://github.com/anujshah1003)'s [Tensorboard-examples](https://github.com/anujshah1003/Tensorboard-examples.git) repository. (For your reference, his README.md also includes a YouTube explanation that you might find useful.)

However, the code in that repository does not support TensorFlow 2.x, so I made some modifications to the `mnist-tensorboard-embeddings-1.py` and `tensorboard.py` files.

Now, after cloning all the files and running TensorFlow, you can access the visualized vector DB results by navigating to `localhost:6006`.
