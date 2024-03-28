[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Cs9oCA6b)
# Final Project EE599 Systems for Machine Learning, Spring 2024
### University of Southern California
### Instructors: Arash Saifhashemi

This repository is intended as a minimal example to load Llama 2 models and run inference, which is based on the offical implementation of [llama](https://github.com/meta-llama/llama) inference from Facebook.
The modifications made are as follows:

* Remove dependencies on the `fairscale` and `fire` packages.
* Remove chat completion feature.
* Reorganize the code structure of the generation feature. Now `Generation` class is the base class of the llama model.
* Remove `logit` related features from `Generation` class.

## Quick Start
Install `torch` and `sentencepiece` packages.

Change `model_path` and `tokenizer_path` in `inference.py`

```
python inference.py
```

## Deadline:
* Phase 1: **Apr 12th, 11:59PM**
* Phase 2: **Apr 22th, 11:59PM**
* Phase 3: **May 9th, 11:59PM**
