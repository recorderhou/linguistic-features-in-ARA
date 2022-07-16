# linguistic-features-in-ARA

To analyze the effect of some simple linguistic features on ARA.

To extract features:



Run these code by:

**bert_with_feature.py**
python bert_with_feature.py --dataset_path=[ENTER YOUR DATASET PATH] --class_num=[THE SIZE OF YOUR LABEL SET] --train_steps=[the steps between two evaluations] --data_portion=[] --max_seq_len=[max length that the PLM could process] --test_time=1

**bert_without_feature.py**
python bert_with_feature.py --dataset_path=[ENTER YOUR DATASET PATH] --class_num=[THE SIZE OF YOUR LABEL SET] --train_steps=[the steps between two evaluations] --data_portion=[] --test_time=1

**longformer.py**
python longformer.py --dataset=[your dataset name] --class_num=[the size of your label set]

Please make sure that your datset is in './[dataset_name]'

### Datasets
RAZ and Newsela is unavaliable due to privacy reasons.
OneStopEnglish: https://github.com/nishkalavallabhi/OneStopEnglishCorpus

