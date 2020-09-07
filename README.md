# Towards Unique and Informative Captioning of Images

Code for the ECCV paper:

[Towards Unique and Informative Captioning of Images](http://www.ecva.net/papers/eccv_2020/papers_ECCV/html/350_ECCV_2020_paper.php)

Zeyu Wang, Berthy Feng, Karthik Narasimhan, Olga Russakovsky

```
@inproceedings{wang2020spiceu,
  title={Towards Unique and Informative Captioning of Images},
  author={Zeyu Wang and Berthy Feng and Karthik Narasimhan and Olga Russakovsky},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020},
}
```

## Requirements
- java 1.8.0+
- python 3+
- [SPICE](https://github.com/peteanderson80/SPICE)


## Usage

1. Parse the captions using the code from [SPICE](https://github.com/peteanderson80/SPICE) (see SPICE for the input format). Example:

```
java Xmx8G -jar spice-1.0.jar example_input.json -out example_parsed.json -detailed -subset
```

2. Compute SPCIE-U score with `compute_spiceu.py`. Example:

```
python compute_spiceu.py --parsed_input example_parsed.json --uniqueness_dict coco_train_uniqueness_dict.pkl
```

where the *uniqueness_dict* is a dictionary with key as concept and value the corresponding uniqueness score for the concept. In the paper, we calcuate the uniqueness score for a concept *c* using the training set with `uniqueness(c) = # images not containing c / # images total`. But it can be modified based on the specific application.
