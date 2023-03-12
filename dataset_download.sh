#!/opt/homebrew/Cellar/bash/5.2.15/bin/bash
#!/bin/bash

# This script requires that the Google Cloud SDK CLI tool is installed
# this can be done by running the ./setup.sh script (on debian based machines)
# or following the instructions at https://cloud.google.com/sdk/docs/install#deb

function usage {
  echo "Usage: ./dataset_download.sh [-d output_directory] [-b|--band data_band] [-h|--help]"
  echo "Options:"
  echo "  -h                    Show this help message and exit"
  echo "  -b --band <band>      Specify the EM band to use, default is B7: https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research"
  echo "  -d --directory <dir>  Specify the output directory, default is ./data/images"
}

while getopts ":d:b:h-:" opt; do
  case ${opt} in
    d ) output_directory="$OPTARG"
      ;;
    b|--band ) IMGBAND="$OPTARG"
      ;;
    h) usage
      exit 0
      ;;
    - ) case "${OPTARG}" in
          band=* ) IMGBAND="${OPTARG#*=}"
            ;;
          directory=* ) output_directory="${OPTARG#*=}"
            ;;
        esac
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

if [ -z "$IMGBAND" ] ; then
  IMGBAND="B7"
fi

if [ -z "$output_directory" ] ; then
  output_directory="./data/images"
fi


# Download landsat 4 images ending in _"$band".TIF
files_download=0
value="050"

for i in {1..5..40}
do
  #Format the string with the incremented substring
  echo "Downloading images for row gs://gcp-public-data-landsat/LT04/01/$value/"
  gspath="gs://gcp-public-data-landsat/LT04/01/$value/*/"
  gsdirlist=$(gsutil ls -d $gspath)
  for f in $gsdirlist
   do
    array+=($f)
  done
  echo $array

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
        
          # Check if the element ends with the string "_B7.tif"
          if [[ $element == *_199[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]*B7.TIF ]]; then
            
            # Check if the file has already been downloaded
            if [ -f "$output_directory/$(basename $element)" ]; then
              continue

            else
              gsutil -m cp "$element" "$output_directory"
              files_download=$((files_download + 1))
            fi
          fi
          # DOwnload the corresponding .txt file
          # *_MTL\.txt$ 
          if [[ $element == *_199[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]*MTL.txt ]]
          then
          echo "checking $element"
            # Check if the file has already been downloaded
            if [ -f "$output_directory/$(basename $element)" ]; then
              continue
            else
              gsutil -m cp "$element" ./data/metadata
            fi
          fi
      done

  done

  # Increment the value substring by 1
  value=$(printf '%03d' $((10#$value + 1)))

done

# Report the number of files downloaded
echo "Downloaded $files_download files"
