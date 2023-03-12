#!/bin/bash

# This script requires that the Google Cloud SDK CLI tool is installed
# this can be done by running the ./setup.sh script (on debian based machines)
# or following the instructions at https://cloud.google.com/sdk/docs/install#deb

echo ""
echo "--- Landsat Imageset Download ---"
echo "Note: This script can take a long time to run"
echo ""

function usage {
  echo "Usage: ./dataset_download.sh [-d output_directory] [-b data_band] [-s <WidthxHeight>] [-h]"
  echo "Options:"
  echo "  -h                  Show this help message and exit"
  echo "  -b <band>           Specify the EM band to use, default is B7: https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research"
  echo "  -s <WidthxHeight>   Specify the output image size, default is 512x512"
}

while getopts ":s:b:h-:" opt; do
  case ${opt} in
    b ) IMGBAND="$OPTARG"
      ;;
    s ) SIZE="$OPTARG"
      ;;
    h) usage
      exit 0
      ;;
    \? ) echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    : ) echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))


trap ctrl_c INT
function ctrl_c() {
  echo ""
  echo "Ending download, Total files downloaded: $files_download"
  if [[ ! -z "$filename" ]]; then
    convert "$IMAGE_DIRECTORY/$filename" -sample "$SIZE" "$IMAGE_DIRECTORY/$filename" 2>/dev/null
  fi
  exit 1
}


if [ -z "$IMGBAND" ] ; then
  IMGBAND="B7"
fi

if [ -z "$SIZE" ] ; then
  SIZE="512x512"
fi


BATCH_NAME="${IMGBAND}_${SIZE}"
IMAGE_DIRECTORY="./data/images/$BATCH_NAME"
METADATA_DIRECTORY="./data/metadata/$BATCH_NAME"


if [ ! -e "./data/images" ]; then
  mkdir -p "./data/images"
fi

if [ ! -e "./data/metadata" ]; then
  mkdir -p "./data/metadata"
fi

if [ ! -e "./data/images/$BATCH_NAME" ]; then
  mkdir -p "./data/images/$BATCH_NAME"
fi

if [ ! -e "./data/metadata/$BATCH_NAME" ]; then
  mkdir -p "./data/metadata/$BATCH_NAME"
fi


echo "Band: $IMGBAND"
echo "Size: $SIZE"
echo "Batch Name: $BATCH_NAME"
echo "Image Directory: $IMAGE_DIRECTORY"
echo "Metadata Directory: $METADATA_DIRECTORY"
echo ""


filename=""
files_download=0
value="180"

for i in {180..230}
do
  #Format the string with the incremented substring
  echo "Downloading images for row gs://gcp-public-data-landsat/LT04/01/$value/"
  gspath="gs://gcp-public-data-landsat/LT04/01/$value/*/"
  gsdirlist=$(gsutil ls -d $gspath)
  readarray -t array <<< "$gsdirlist"

  # Filter out subdirectories ending in $folder$
  subdirectories=()
  for row in "${array[@]}"
    do
      if [[ $row != *"\$folder\$" ]]; then
        subdirectories+=("$row")
      fi
  done

  subdirectories=("${subdirectories[@]:1}")

  # Find all of the images ending in <IMGBAND>.TIF in the subdirectories
  for row in "${subdirectories[@]}"
    do
      # Create a new base URL for the subdirectory
      basepath="$row**"
      echo "Searching for band $IMGBAND images at $basepath"
      gsdirlist=$(gsutil ls -d $basepath)
      mapfile -t basearray <<< "$gsdirlist"

      for element in "${basearray[@]}"
        do
          # Check if the element ends with the string "_<BAND>.tif"
          if [[ "$element" == *_"$IMGBAND".TIF ]]; then
            filename=$(basename $element)
            gsutil -m cp -n "$element" "$IMAGE_DIRECTORY"
            convert "$IMAGE_DIRECTORY/$filename" -sample "$SIZE" "$IMAGE_DIRECTORY/$filename" 2>/dev/null
            files_download=$((files_download + 1))
          fi
          # Download the corresponding .txt file
          if [[ "$element" == *MTL.txt ]]; then
            gsutil -m cp -n "$element" "$METADATA_DIRECTORY"
          fi
      done
  done

  if [ $files_download -eq 6000 ]; then
    echo "Ending download, Total files downloaded: $files_download"
    exit 0
  fi

  value=$(printf '%03d' $((10#$value + 1)))

done
