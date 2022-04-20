## preprocessing.py

Preprocess corpora: strip target words of their POS tags (if any) and generate train/test sets

#### Arguments:

- `--targets_path`  
  Path to a .txt file with target words
- `--corpora_paths`  
  Paths to corpora separated with ; The entire string with paths has to be surrounded with quotes
- `--corpora_language`  
  "spanish" for this particular task
- `--output_path`  
  Path to a folder for processed files

#### Sample usage:

```
python preprocessing.py --targets_path Data/targets.txt --corpora_paths "Data/old.txt;Data/modern.txt" --corpora_language spanish --output_folder Test
```

## run_mlm.py

Fine-tune a pre-trained BERT model.

#### Key Arguments:

- `--model_name_or_path`  
  name of the model to fine-tune (from https://huggingface.co/models)
- `--train_file`  
  path to the file with train corpus
- `--validation_file`  
  path to the file with validation corpus
- `--do_train`  
  include this argument to perform training (requires --train_file)
- `--do_eval`  
  include this argument to perform validation (requires --validation_file)
- `--output_dir`  
  directory where all the output data will be saved. Specifying an empty directory is recommended. To overwrite
  non-empty existing directory, add --overwrite_output_dir argument
- `--line_by_line`  
  include this argument if you want each line in corpora to be treated as a single training sequence
- `--save_steps` (default=500)  
  Determines how many training sequences will be included in a single checkpoint. The less is the number, the more
  checkpoints will be created
- `--save_total_limit` (default=None)  
  initialize this argument with a number of maximum checkpoints to be created (delete older checkpoints)
- `--max_seq_length`  
  it is highly recommended setting this to 512 for compatibility with BERT

#### Sample usage:

```
python run_mlm.py --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased --train_file Processed/train.txt --validation_file Processed/test.txt --do_train --do_eval --output_dir Processed/MLM --line_by_line --save_steps 5000 --save_total_limit 20 --max_seq_length 512
```

## extract_embeddings.py

Extract embeddings for target words (specific layers) from a fine-tuned model.

#### Arguments:

- `--model_path`  
  path to a .bin PyTorch model
- `--model_name`  
  name of a model used during fine-tuning
- `--targets_path`  
  Path to a .txt file with target words
- `--corpora_paths`  
  Paths to PREPROCESSED corpora separated with ; The entire string with paths has to be surrounded with quotes
- `--output_path`  
  Path to a result file with embeddings
- `--bert_layers`  
  Bert layers to extract (from 0 to 11). Possible ways to provide:
    - separated with commas (0,4,5)
    - separated with a dash (3-5 is the same as 3,4,5)

  If this argument is omitted, all 12 layers will be extracted
- `--concat_layers`  
  Pass this argument if you want to concatenate the layers rather than sum them. Disabled by default
- `--gpu`  
  Pass this argument to use GPU during calculations. Disabled by default
- `--checkpoint_frequency`  
  Specify the number of lines that need to be processed before making a checkpoint (e.g. set to 100000 to make a
  checkpoint every 100000 lines). If this argument is not specified, no checkpoints will be created.

#### Result format:

- The embeddings are saved as a dict where:
    - keys are corpora names and values are dicts where:
        - keys are words and values are numpy arrays with embeddings stacked vertically

For example, calling EMBEDDINGS['Data/ccoha1.txt']['attack'][0] we can get the first embedding for the word 'attack'
from 'ccoha1' corpora. EMBEDDINGS['Data/ccoha1.txt']['attack'][1] is the second embedding, and so on.

#### Sample usages:

```
python extract_embeddings.py --model_path Processed/pytorch_model.bin --model_name dccuchile/bert-base-spanish-wwm-uncased --targets_path targets.txt --corpora_paths Processed/processed1.txt;Processed/processed2.txt --output_path test.pickle
```

```
python extract_embeddings.py --model_path Processed/pytorch_model.bin --model_name dccuchile/bert-base-spanish-wwm-uncased --targets_path targets.txt --corpora_paths Processed/processed1.txt;Processed/processed2.txt --output_path test.pickle --bert_layers 0,5,7-9,11 --concat_layers
```

## predictions.py

Evaluate the embeddings (calculate graded/binary change, calculate Spearman rank if applicable)

#### Arguments:

- `embeddings_path`  
  Path to pickled embeddings
- `targets`  
  Path to a text file with targets. The file may contain target words with gold standard values (in that case Spearman
  rank will be calculated) or simply a list of target words (in that case graded/binary/change/loss will be calculated).
- `metric`  
  A metric to use for calculating semantic change. Two values are supported: `average` or `k_means`.
- `report_path`  
  A full path to the file where report will be saved
- `n_clusters`  
  This argument is only applicable for the `k_means` metric and is ignored for the `average` metric. It is the number of
  clusters for clustering. Default value is 8.

#### Sample usage:

```
python predictions.py --embeddings_path /home/aishein_1/embeddings/dccuchile_bert-base-spanish-wwm-uncased.pickle --targets /home/aishein_1/targets/target_words_evaluation_phase2.txt --metric k_means --n_clusters 28 --report_path /home/aishein_1/results/binary_final_gain_loss/answer/submission.tsv
```