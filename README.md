# Document-Similarity-K-shingles-minhashing-LSH

Introduction

Many big-data problems can be expressed as finding "similar" items. In this project we
will investigate similarities among 21578 documents from a cleanup collection of
documents were made available by Reuters and CGI for research purposes. The
collection appeared in 1987 and after processing in 1996 the data set had the form we
know today with 21578 text categorization collection. As the name indicates, this
collection contains 21578 text documents from Reuters Ltd. Το be more precise, the
collection consists of 22 data files, an SGML DTD file describing the data file format,
and six files describing the categories used to index the data. Each of the first 21 files
(reut2-000.sgm through reut2-020.sgm) contain 1000 documents, while the last
(reut2-021.sgm) contains 578 documents.

The AIM of this Assignment is to discover relationships between these texts, using kShingles,
Jaccard similarities through Minhashing and Locality Sensitive Hashing. We
are interested to investigate how similar the texts are. For this purpose we think data
as "Sets" of "Strings" and convert shingles into minhash signatures.
For the whole analysis we used Python 2.7. For graphs we used Microsoft excel.
In more detail

Dataset

The documents used for this project appeared on the Reuters newswire in 1987. In
1990, the documents were made available by Reuters and CGI for research purposes
at the University of Massachusetts at Amherst. Formatting of the documents and
production of associated data files was done in 1990. Further formatting and data file
production was done in 1991 and 1992 at the Center for Information and Language
Studies, University of Chicago. This version of the data was made available for
anonymous FTP as "Reuters22173, Distribution 1.0" in January 1993. From 1993
through 1996, Distribution 1.0 was hosted at a succession of FTP sites maintained by
the Center for Intelligent Information Retrieval of the Computer Science Department
at the University of Massachusetts at Amherst. At the ACM SIGIR '96 conference in
August, 1996 a group of text categorization researchers discussed how published
results on Reuters-22173 could be made more comparable across studies. It was
decided that a new version of collection should be produced with less ambiguous
formatting, and including documentation carefully spelling out standard methods of
using the collection. The opportunity would also be used to correct a variety of
typographical and other errors in the categorization and formatting of the collection.
One result of the re-examination of the collection was the removal of 595 documents
which were exact duplicates (based on identity of timestamps down to the second) of
other documents in the collection. The new collection (which one we used in analysis)
therefore has only 21578 documents.

The dataset can be found here 
https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection
and should be extracted in the /data directory

K shingles input
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/kshingles.jpg)

3 shingles output
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/shingles.PNG)

Minhashing input
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/hashfunctions.jpg)

Minhashing output
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/minhashing.jpg)

LSH input
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/lsh.jpg)

Shingle similarity 
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/jaccard%20sim.jpg)

Signature similarity 
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/shingle%20sim.jpg)

LSH similarity
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/lsh%20sim.jpg)

Time consumption
![alt text](https://github.com/evagian/Document-similarity-K-shingles-minhashing-LSH-python/blob/master/data/doc/time.png)


