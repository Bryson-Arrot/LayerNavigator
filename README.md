# LayerNavigator

This repository provides the **official implementation** and **data** for the main results presented in our paper. The code is designed to be as **simple and intuitive** as our proposed method.

---

## 🧩 Environment Setup

Our experiments were conducted under the following environment.
If your current setup cannot run the code smoothly, you may refer to these versions as a reference configuration:

* **Python version:** 3.8.10
* **CUDA version:** 11.3
* **Other dependencies:** see [`requirements.txt`](./requirements.txt)

---

## 🚀 Quick Start

We provide complete code and data to reproduce the **main results** from our paper.
You can run everything with just **two steps**:

1. **Set model path**
   In the first line of `globalenv.py`, modify the path to your model checkpoint.

2. **Run the main script**

   ```bash
   python main.py
   ```

That’s it — this will execute the experiments and reproduce the main results in the paper.

---

## 📁 File Descriptions

| File                   | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **`dataset.py`**       | Implements dataset loading and tokenization. Supports training data (for collecting activations and further vectors) as well as validation/test data (for evaluating model behavior). |
| **`model_wrapper.py`** | Wraps the base model to extract internal activations and adding steering vectors during inference. |
| **`get_vec.py`**       | Implements steering vector extraction using **Mean Difference** or **PCA** methods, and saves activations used during computation. |
| **`get_score.py`**     | **Core of LayerNavigator.** Computes the *steerability score* for each layer based on activations and steering vectors. |
| **`strategy.py`**      | Selects target layers for steering based on the computed scores. Also includes methods for adjusting the weighting between *discriminability* and *consistency* scores. |
| **`get_results.py`**   | Implements methods for obtaining model logits and text generation outputs. |

---

## 💡 Note

We have made every effort to ensure the code is **concise, readable, and easy to extend**—just like the **LayerNavigator** method itself.
