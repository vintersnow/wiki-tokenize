# Tokenize Wikipedia Dump

Tokenize the Wikipedia dump.
Extact all sentences and count tokens (words) frequences.

# Requirements

## Python package

- nltk

## Data

Wikipedia dump
Change to json files using [wikiextractor](https://github.com/attardi/wikiextractor)

# Usage

```
python --data_path 'path/to/data_dirs' --out_path 'path/to/output_dir'
```

# Output

- output-{0-3}.txt
	- Format: one sentence for each line.
- vocab
	- Format: `{word} {frequences}`
