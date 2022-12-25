# torch-utils
a collection of pytorch functions and models to get the sh1t done.
to use it inside of a python notebook:


```python
try:
    from torch_utils import data_setup, engine
except:
    print("installing  files from 'https://github.com/yonatan-co/torch_snippets.git'")
    !git clone https://github.com/yonatan-co/torch_utils.git
    !rm torch_utils/README.md
    from torch_utils import data_setup, engine
```
