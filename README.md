# sparql_translater
SPAQRL neural translater for Human Robot Interactions.

This repository contains the ressources to train neural networks for seq2seq, and in particular english to SPARQL. SPARQL is a semi-specified language used to query RDF graphs (which are a type of relationnal database).

## Install dependencies

To install the dependencies just use the following command
```shell
pip install -r requirements.txt
```

## Multilabel classification

This model is use to classify labels about the sentence, which helps to revolve ambiguities or perspective related relations.

```shell
cd multilabel_classif
./download_fasttext.sh
python generate_data.py
python train_and_evaluate.py
```

## Neural Translation

```shell
cd seq2seq
./download_glove.sh
python generate_data.py
python build_vocab.py
train_and_evaluate.sh
```
