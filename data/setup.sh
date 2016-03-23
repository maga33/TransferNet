# download coco, voc, imagesets
# imagesets
wget http://cvlab.postech.ac.kr/research/transfernet/data/imagesets.tar.gz
tar -zxvf imagesets.tar.gz
rm -rf imagesets.tar.gz

# voc
wget http://cvlab.postech.ac.kr/research/transfernet/data/VOC2012_SEG_AUG.tar.gz
tar -zxvf VOC2012_SEG_AUG.tar.gz
rm -rf VOC2012_SEG_AUG.tar.gz

# coco
wget http://cvlab.postech.ac.kr/research/transfernet/data/MSCOCO.tar.gz
tar -zxvf MSCOCO.tar.gz
rm -rf MSCOCO.tar.gz

