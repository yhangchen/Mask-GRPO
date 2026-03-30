#!/bin/bash
# Rename WISE images from 0001_0.png -> ID1_0.jpg
set -euo pipefail

DIR="${1:-/data/banyuanhao/Mask-GRPO_outputs/WISE/base_model}"

count=0
for f in "$DIR"/[0-9][0-9][0-9][0-9]_*.png; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    # Extract prompt_id (remove leading zeros) and gen_idx
    prompt_id=$(echo "$base" | sed 's/^0*//' | cut -d_ -f1)
    gen_idx=$(echo "$base" | cut -d_ -f2 | sed 's/\.png$//')
    new_name="ID${prompt_id}_${gen_idx}.jpg"
    mv "$f" "$DIR/$new_name"
    count=$((count + 1))
done

echo "Renamed $count files in $DIR"
