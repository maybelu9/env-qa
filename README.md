# Env-QA: A Video QA Benchmark for Comprehensive Understanding of Dynamic Environments
[Webpage](https://envqa.github.io/) • [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Env-QA_A_Video_Question_Answering_Benchmark_for_Comprehensive_Understanding_of_ICCV_2021_paper.pdf)

This repository provides the code for dataloader and evaluation code for Env-QA dataset.

## Requirements
To install requirements, run:
```
pip install -r requirements.txt
```

## Dataloader
Please download all annotations (`train_full_question.json`, `val_full_question.json`, `test_full_question.json`, `env_qa_video_annotations_v1.json`, `env_qa_full_predicted_segment.json`, `dictionaries.pkl`, `dict_object_name.json`, `all_instructions.json`) and features (`env_qa_objects.h5`, `env_qa_frame_obj_cls.h5`), and put them under the `data/` folder.
Please see the webpage (https://envqa.github.io/) to download the dataset.

We provide a start code on `dataloader_evaluater.ipynb`

You can follow the guidance to organize these files to load the dataset.

## Evaluation
We also provide an example in `dataloader_evaluater.ipynb`. Please see the file to use our evaluation code.


## Citation 
If you found this work useful, consider citing our papers as followed:
```
@inproceedings{Gao_2021_ICCV,
  title={Env-QA: A Video Question Answering Benchmark for Comprehensive Understanding of Dynamic Environments,
  author={Gao, Difei and Wang, Ruiping and Bai, Ziyi and Chen, Xilin},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
  month={October},
  year={2021},
  pages = {1675-1685}
}
```
