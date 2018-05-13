# Textual Outlier Detection


### To Run:
```bash
export PYTHONHASHSEED=0
export SPARK_YARN_USER_ENV=0

spark-submit --py-files=lib.zip main.py \
    --input_hfs_path='reddit_sarcasm_small.txt' \
    --output_hfs_path='reddit_sarcasm_small_outliers.txt' \
    --config_json_path='configs/default.json'
```

### To build:

```bash
source build.sh
```
