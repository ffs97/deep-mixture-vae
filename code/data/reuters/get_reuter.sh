#!/bin/sh
wget http://jmlr.org/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt0.dat.gz
gunzip lyrl2004_vectors_test_pt0.dat.gz
wget http://jmlr.org/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt1.dat.gz
gunzip lyrl2004_vectors_test_pt1.dat.gz
wget http://jmlr.org/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt2.dat.gz
gunzip lyrl2004_vectors_test_pt2.dat.gz
wget http://jmlr.org/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt3.dat.gz
gunzip lyrl2004_vectors_test_pt3.dat.gz
wget http://jmlr.org/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz
gunzip lyrl2004_vectors_train.dat.gz
wget http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
wget https://ndownloader.figshare.com/files/5976048
mv 5976048 rcv1-v2.topics.qrels.gz
gunzip rcv1-v2.topics.qrels.gz
