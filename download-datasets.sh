# ## Flick8k
# wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
# unzip -q flickr8k.zip -d ./flickr8k
# rm flickr8k.zip
# echo "Downloaded Flickr8k dataset successfully."

## Flickr30k
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip
rm flickr30k_part00 flickr30k_part01 flickr30k_part02
unzip -q flickr30k.zip -d ./flickr30k
rm flickr30k.zip
echo "Downloaded Flickr30k dataset successfully."

## MSCOCO
# wget "http://images.cocodataset.org/zips/train2014.zip"
# wget "http://images.cocodataset.org/zips/val2014.zip"
# unzip -q train2014.zip -d ./mscoco
# unzip -q val2014.zip -d ./mscoco
# wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
# unzip -q annotations_trainval2014.zip -d ./mscoco
# rm train2014.zip val2014.zip annotations_trainval2014.zip