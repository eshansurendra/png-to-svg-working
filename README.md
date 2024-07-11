# png-to-svg-working

## TinyLLaVA_Factory approach

### Original dataset format

```
{"wikicommons_url":,
"query_word": <plot/geometry/graph..>,
"filename": <filename in our local folder with pngs>,
"complete_svg_code": ...,
"clean_svg_code" ...}
```

### Finetune TinyLLaVA with Custom Datasets

As per the official documentation here: 
https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/main/CUSTOM_FINETUNE.md

We converted the dataset data to a JSON file of a List of all samples. Sample metadata contains id (a unique identifier), image (the path to the image), and conversations (the conversation data between human and AI).

```
[
  {
        "id": "meiKqU2auAVK2vrtLhKGoJ",
        "image": "pokemon/image/meiKqU2auAVK2vrtLhKGoJ.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nProvide a brief description of the given image."
            },
            {
                "from": "gpt",
                "value": "a drawing of a green pokemon with red eyes"
            }
        ]
    }
]
```
For our dataset

```
[
  {
        "id": "meiKqU2auAVK2vrtLhKGoJ",
        "image": "images/<filename in our local folder with pngs>",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n convert to SVG code."
            },
            {
                "from": "gpt",
                "value": <clean svg code>.
            }
        ]
    }
]
```
The conversion was done using a Kaggle notebook since colab faced some memory limitations and repeatedly failed to execute.

Kaggle notebook link - https://www.kaggle.com/code/eshansurendra/tinyllava-custom-datasets-preparation/edit

Published that dataset for further ease of usage.


### Finetune TinyLLaVA using LORA

Used kaggle notebooks to finetune but the attempt failed with GPU run out of memory 
Then tried with TPU memory is sufficient but the DeepSpeed currently not supporting for TPU


