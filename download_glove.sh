
#!/bin/bash
# This script install the Glove dataset
echo "Start downloading the Glove dataset... this may take a while..."
mkdir -p models/word_embeddings
cd ./models/word_embeddings
mkdir glove && cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
cd ..
echo "Bye bye !"
