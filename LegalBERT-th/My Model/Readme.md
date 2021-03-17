# LegalBERT-th

#### LegalBERT-th presents the Legal Thai pre-trained model based on the BERT-th of ThALKeras structure. It is now available to download.
- [LegalBERT - th](https://drive.google.com/file/d/1pU7FHnhgCmZZfL8gosI7XuZBpbsvElil/view?usp=sharing) 

- [save_model.pb](https://drive.google.com/drive/folders/1ed933Vk6k6uEtypIP7a9PG6oKJcuKnKR?usp=sharing) 


## Preprocessing

### Data Source

Training data for BERT-th come from [the latest article dump of Thai Wikipedia](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) on November 2, 2018. The raw texts are extracted by using [WikiExtractor](https://github.com/attardi/wikiextractor).

### Sentence Segmentation

Input data need to be segmented into separate sentences before further processing by BERT modules. Since Thai language has no explicit marker at the end of a sentence, it is quite problematic to pinpoint sentence boundaries. To the best of our knowledge, there is still no implementation of Thai sentence segmentation elsewhere. So, in this project, sentence segmentation is done by applying simple heuristics, considering spaces, sentence length and common conjunctions.

After preprocessing, the training corpus consists of approximately 2 million sentences and 40 million words (counting words after word segmentation by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)). The plain and segmented texts can be downloaded **[`here`](https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing)**.

## Tokenization

BERT uses [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) as a tokenization mechanism. But it is Google internal, we cannot apply existing Thai word segmentation and then utilize WordPiece to learn the set of subword units. The best alternative is [SentencePiece](https://github.com/google/sentencepiece) which implements [BPE](https://arxiv.org/abs/1508.07909) and needs no word segmentation.

In this project, we adopt a pre-trained Thai SentencePiece model from [BPEmb](https://github.com/bheinzerling/bpemb). The model of 25000 vocabularies is chosen and the vocabulary file has to be augmented with BERT's special characters, including '[PAD]', '[CLS]', '[SEP]' and '[MASK]'. The model and vocabulary files can be downloaded **[`here`](https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/view?usp=sharing)**.

`SentencePiece` and `bpe_helper.py` from BPEmb are both used to tokenize data. `ThaiTokenizer class` has been added to BERT's `tokenization.py` for tokenizing Thai texts.

## Pre-training LegalBERT - th

Dataset for Pre - training downloaded 
> Small [dummylaw_sentseg](https://drive.google.com/file/d/1HMrssJmVlIYMajQ6XGMbU6XMa0dHdneI/view?usp=sharing)

> large [lawtext_sentseg](https://drive.google.com/file/d/1TisI4yIvvE2y6a_C_w4vAOaHYlDyS4Uf/view?usp=sharing)


The data can be prepared before pre-training by using this script.

```shell
export BPE_DIR=/path/to/bpe
export TEXT_DIR=/path/to/dummylaw_sentseg or lawtext_sentseg\ 
export DATA_DIR=/path/to/Output

python bert/create_pretraining_data.py \
  --input_file=$TEXT_DIR \
  --output_file=$DATA_DIR/tf_examples.tfrecord \
  --vocab_file=$BPE_DIR/th.wiki.bpe.op25000.vocab \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --thai_text=True \
  --spm_file=$BPE_DIR/th.wiki.bpe.op25000.model
```

Then, the following script can be run to learn a model from scratch.

```shell
export DATA_DIR=/path/to/tf_examples.tfrecord of Pre-training
export BERT_BASE_DIR=/path/to/BERT - th
export BERT_LEGAL_DIR=/path/to/Output


python bert/run_pretraining.py \
  --input_file=$DATA_DIR/tf_examples.tfrecord \
  --output_dir=$BERT_LEGAL_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=5e-5 \
```


## Downstream Classification Tasks

### legaldoc

Thai Law Dataset of Webboaed downloaded 
[Legal_Dataset](https://drive.google.com/drive/folders/1ZmlXEewbch-SpDscnrgJXzc14JK8oz8s?usp=sharing) 

Raw Dataset 
[Question & Answer AND Dummy Dataset](https://drive.google.com/drive/folders/11D9DLQKtesjDz1-Lm314c-AWY2sZFEgR?usp=sharing)


Afterwards, the legaldoc task can be learned by using this script.

```shell
export BPE_DIR=/path/to/bpe
export LAW_DIR_DIR=/path/to/Legal_Dataset
export OUTPUT_DIR=/path/to/output
export BERT_BASE_DIR=/path/to/BERT - th
export BERT_LEGAL_DIR=/path/to/LegalBERT - th  >>  to model.ckpt-20

python bert/law_classifier.py \
  --task_name=legaldoc \
  --do_train=true \
  --do_eval=true \
  --do_export=true \
  --data_dir=$LAW_DIR \
  --vocab_file=$BPE_DIR/th.wiki.bpe.op25000.vocab \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LEGAL_DIR/model.ckpt-20 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --spm_file=$BPE_DIR/th.wiki.bpe.op25000.model
```


<!-- Use html table because github markdown doesn't support colspan -->
<table>
  <tr>
    <td colspan="2" align="center"><b>LegalBERT - th</b></td>
  </tr>
  <tr>
    <td align="center">Accuracy (%)</td>
  </tr>
    <td align="center"><b>92</b></td>
</table>


#### GitHub my Project (Code scraping data: Thai law, Dataset Development)
- [here](https://github.com/WiratchawaKannika/LegalDoc_project4) 

#### GitHub for Web Application (Prototype) connect with API for Identifying Type of Law of Post in Legal Webboards
- [here](https://github.com/WiratchawaKannika/WebApp_LegalBERT-th) 

#### WebPage detail of LegalBERT-th Project
- [here](https://wiratchawakannika.github.io/LegalDoc_NLP/) 
