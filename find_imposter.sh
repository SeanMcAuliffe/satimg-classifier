cd data/metadata/
for f in *; do
    if [ ! -f "../images_norm/${f:0:36}*" ]; then
        echo "Extra metadata file: $f"
        break
    fi
done
