# TriplesExtraction

This repository runs two approaches to extract knowledge triples from the text. The first approach is based on dependency parsing while the second approach uses Bert token classification to classify the knowledge triples. Currently this repository only supports 20 News Dataset. 


## Features

- Downloads and cleans the data.
- Extracts triples.
- Saves extracted triples in data/20NewsGroups.csv

## Running

To do the extraction via dependency parse, use 'dep' mode in command line.
```sh
python3 extract.py --mode dep
```
The script will extract the triples and will save them in `data/20NewsGroups.csv` in 'triples' column.

For extracting triples doing inference on transformer based model, use 'bert' mode in command line.

But, beforehand, please make sure the following points are in check.
 - The checkpoint has been downloaded from the google drive link (provided in the email) and the `config.py` file has the location of checkpoint. 
 

```sh
python3 extract.py --mode bert
```

## Note

The GPU based training can be done using `Colab_training: Triples.ipynb` notebook. 
The GPU based k-fold inference can be done using `Colab_inference: Triples.ipynb` notebook. 

To use these notebooks, please point the required data/checkpoints in the Config->Globals section.

Thank you.
