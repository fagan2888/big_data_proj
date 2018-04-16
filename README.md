# Textual Outlier Detection


### To Run:
```bash
export PYTHONHASHSEED=0
export SPARK_YARN_USER_ENV=0

spark-submit main.py \
    --input_hfs_path='reddit_sarcasm_small.txt' \
    --output_hfs_path='reddit_sarcasm_small_outliers.txt' \
    --config_json_path='default_config.json'
```

### To build:
```bash
zip zipped.zip filters/*
```
