from bs4 import BeautifulSoup
import sys
import os.path
import string
import os
import re
import random
import time
import binascii

documents = []
printedbodies = {}
print 'Reading files'
print 'Please wait...'
t0 = time.time()

data = ''

for file in os.listdir("data/"):
    if file.endswith(".sgm"):
        filename = os.path.join("data", file)

        f = open(filename, 'r')
        data = data + f.read()

print 'Reading data took %.2f sec.' % (time.time() - t0)

print 'Transforming data...'
t0 = time.time()
soup = BeautifulSoup(data, "html.parser")
bodies = soup.findAll('body')
i = 0
for body in bodies:
    printedbodies[i] = body
    documents.append(
        re.sub(' +', ' ', str(body).replace("<body>", "").replace("</body>", "").translate(None, string.punctuation)
               .replace("", "").replace("\n", " ").lower()))
    i = i + 1

print 'Transforming data took %.2f sec.' % (time.time() - t0)

print 'The number of documents read was: ' + str(len(documents))

i = 0
d = {}

t = {}
t0 = time.time()
for value in documents:


    # create a dictionary where key=docid and value=document text
    d[i] = value
    # split text into words
    d[i] = re.sub("[^\w]", " ", d[i]).split()

    # remove rows with empty values from dictionary d
    if d[i]:
        i = i + 1
    else:
        del d[i]
        del body[i]

# =============================================================================
#               Convert Documents To Sets of Shingles
# =============================================================================

docsAsShingleSets = {}

docNames = []

totalShingles = 0
shingleNo = 0
######Ask user to give a value for k
while True:
    try:
        shingle_size = int(raw_input("Please enter k value for k-shingles: "))
    except ValueError:
        print("Your input is not valid. Give a positive natural number > 0...")
        continue
    if shingle_size <= 0:
        continue
    else:
        break

print "Shingling articles..."

t0 = time.time()
# loop through all the documents
for i in range(0, len(d)):

    # Read all of the words (they are all on one line)
    words = d[i]

    # Retrieve the article ID
    docID = i

    # Maintain a list of all document IDs.
    docNames.append(docID)

    # 'shinglesInDoc' will hold all of the unique shingles present in the
    # current document. If a shingle ID occurs multiple times in the document,
    # it will only appear once in the set.

    # keep word shingles
    shinglesInDocWords = set()

    # keep hashed shingles
    shinglesInDocInts = set()

    shingle = []
    # For each word in the document...
    for index in range(len(words) - shingle_size + 1):
        # Construct the shingle text by combining k words together.
        shingle = words[index:index + shingle_size]
        shingle = ' '.join(shingle)

        # Hash the shingle to a 32-bit integer.
        crc = binascii.crc32(shingle) & 0xffffffff

        if shingle not in shinglesInDocWords:
            shinglesInDocWords.add(shingle)
        # Add the hash value to the list of shingles for the current document.
        # Note that set objects will only add the value to the set if the set
        # doesn't already contain it.

        if crc not in shinglesInDocInts:
            shinglesInDocInts.add(crc)
            # Count the number of shingles across all documents.
            shingleNo = shingleNo + 1
        else:
            del shingle
            index = index - 1

    # Store the completed list of shingles for this document in the dictionary.
    docsAsShingleSets[docID] = shinglesInDocInts

totalShingles = shingleNo

print 'Total Number of Shingles', shingleNo
# Report how long shingling took.
print '\nShingling ' + str(len(docsAsShingleSets)) + ' docs took %.2f sec.' % (time.time() - t0)

print '\nAverage shingles per doc: %.2f' % (shingleNo / len(docsAsShingleSets))

# =============================================================================
#                     Define Triangle Matrices
# =============================================================================

# Define virtual Triangle matrices to hold the similarity values. For storing
# similarities between pairs, we only need roughly half the elements of a full
# matrix. Using a triangle matrix requires less than half the memory of a full
# matrix. Using a triangle matrix requires less than half the memory of a full
# matrix, and can protect the programmer from inadvertently accessing one of
# the empty/invalid cells of a full matrix.

# Calculate the number of elements needed in our triangle matrix
numElems = int(len(docsAsShingleSets) * (len(docsAsShingleSets) - 1) / 2)

# Initialize two empty lists to store the similarity values.
# 'JSim' will be for the actual Jaccard Similarity values.
# 'estJSim' will be for the estimated Jaccard Similarities found by comparing
# the MinHash signatures.
JSim = [0 for x in range(numElems)]
estJSim = [0 for x in range(numElems)]


# Define a function to map a 2D matrix coordinate into a 1D index.
def getTriangleIndex(i, j):
    # If i == j that's an error.
    if i == j:
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    # If j < i just swap the values.
    if j < i:
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array.
    # This fancy indexing scheme is taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # But I adapted it for a 0-based index.
    # Note: The division by two should not truncate, it
    #       needs to be a float.
    k = int(i * (len(docsAsShingleSets) - (i + 1) / 2.0) + j - i) - 1

    return k


######Ask user to give a value for hash functions to be used
while True:
    try:
        numHashes = int(raw_input("\nPlease enter how many hash functions you want to be used: "))
    except ValueError:
        print("Your input is not valid. Give a positive natural number > 0...")
        continue
    if numHashes <= 0:
        continue
    else:
        break

print '\nGenerating random hash functions...'

# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================
from time import clock

# Time this step.
t0 = time.time()


# https://www.codeproject.com/Articles/691200/Primality-test-algorithms-Prime-test-The-fastest-w
# check if integer n is a prime
# probabilistic method, much faster than usual priminality tests
def MillerRabinPrimalityTest(number):
    '''
    because the algorithm input is ODD number than if we get
    even and it is the number 2 we return TRUE ( spcial case )
    if we get the number 1 we return false and any other even
    number we will return false.
    '''
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False

    ''' first we want to express n as : 2^s * r ( were r is odd ) '''

    ''' the odd part of the number '''
    oddPartOfNumber = number - 1

    ''' The number of time that the number is divided by two '''
    timesTwoDividNumber = 0

    ''' while r is even divid by 2 to find the odd part '''
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber / 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    '''
    since there are number that are cases of "strong liar" we
    need to check more then one number
    '''
    for time in range(3):

        ''' choose "Good" random number '''
        while True:
            ''' Draw a RANDOM number in range of number ( Z_number )  '''
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        ''' randomNumberWithPower = randomNumber^oddPartOfNumber mod number '''
        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        ''' if random number is not 1 and not -1 ( in mod n ) '''
        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            # number of iteration
            iterationNumber = 1

            ''' while we can squre the number and the squered number is not -1 mod number'''
            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                ''' squre the number '''
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)

                # inc the number of iteration
                iterationNumber = iterationNumber + 1
            '''
            if x != -1 mod number then it because we did not found strong witnesses
            hence 1 have more then two roots in mod n ==>
            n is composite ==> return false for primality
            '''
            if (randomNumberWithPower != (number - 1)):
                return False

    ''' well the number pass the tests ==> it is probably prime ==> return true for primality '''
    return True


# Record the total number of shingles
i = 1
# find first prime which is higher than the total number of shingles
# print 'Total number of shingles = ', shingleNo
while not MillerRabinPrimalityTest(shingleNo + i):
    i = i + 1
print 'Next prime = ', shingleNo + i

maxShingleID = shingleNo
nextPrime = shingleNo + i


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than shingleNo.

# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the
# list.
def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID)

            # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

    return randList


# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

print '\nGenerating MinHash signatures for all documents...'

# List of documents represented as signature vectors
signatures = []

# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.
# For each document...
for docID in docNames:

    # Get the shingle set for this document.
    shingleIDSet = docsAsShingleSets[docID]

    # The resulting minhash signature for this document.
    signature = []

    # For each of the random hash functions...
    for i in range(0, numHashes):

        # For each of the shingles actually in the document, calculate its hash code
        # using hash function 'i'.

        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
        # the maximum possible value output by the hash.
        minHashCode = nextPrime + 1

        # For each shingle in the document...
        for shingleID in shingleIDSet:
            # Evaluate the hash function.
            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

            # Track the lowest hash code seen.
            if hashCode < minHashCode:
                minHashCode = hashCode

        # Add the smallest hash code value as component number 'i' of the signature.
        signature.append(minHashCode)

    # Store the MinHash signature for this document.
    signatures.append(signature)

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)

print "\nGenerating MinHash signatures took %.2fsec" % elapsed

numDocs = len(signatures)

######Ask user to choose a document
while True:
    try:
        docid = int(raw_input(
            "Please enter the document id you are interested in. The valid document ids are 1 - " + str(
                numDocs) + ": "))
    except ValueError:
        print("Your input is not valid.")
        continue
    if docid <= 0 or docid > numDocs:
        print ("Your input is out of the defined range...")
        continue
    else:
        break

######Ask user to give desired number of neighbors
while True:
    try:
        neighbors = int(raw_input("Please enter the number of closest neighbors you want to find... "))
    except ValueError:
        print("Your input is not valid.")
        continue
    if neighbors <= 0:
        continue
    else:
        break

# =============================================================================
#                 Calculate Jaccard Similarities
# =============================================================================
# In this section, we will directly calculate the Jaccard similarities by
# comparing the sets. This is included here to show how much slower it is than
# the MinHash approach.

# Calculating the Jaccard similarities gets really slow for large numbers
# of documents.
from decimal import *

# if True:

print "\nCalculating Jaccard Similarities of Shingles..."

# Time the calculation.
t0 = time.time()

s0 = len(docsAsShingleSets[0])
# For every document pair...
i = docid

# Print progress every 100 documents.
if (i % 100) == 0:
    print "  (" + str(i) + " / " + str(len(docsAsShingleSets)) + ")"

# Retrieve the set of shingles for document i.
s1 = docsAsShingleSets[docNames[i]]
neighbors_of_given_documentSHINGLES = {}
fp = []
tp = []

for j in range(0, len(docsAsShingleSets)):
    if j != i:
        # Retrieve the set of shingles for document j.
        s2 = docsAsShingleSets[docNames[j]]

        # Calculate and store the actual Jaccard similarity.
        JSim[getTriangleIndex(i, j)] = (len(s1.intersection(s2)) / float(len(s1.union(s2))))
        percsimilarity = JSim[getTriangleIndex(i, j)] * 100
        if (percsimilarity > 0):
            # Print out the match and similarity values with pretty spacing.
            print "  %5s --> %5s   %.2f%s   " % (docNames[i], docNames[j], percsimilarity, '%')
            neighbors_of_given_documentSHINGLES[j] = percsimilarity

sorted_neigborsSHINGLES = sorted(neighbors_of_given_documentSHINGLES.items(), key=lambda x: x[1], reverse=True)

print 'Comparing Shingles ...'
print "The " + str(neighbors) + " closest neighbors of document " + str(docNames[i]) + " are:"
for i in range(0, neighbors):
    if i >= len(sorted_neigborsSHINGLES):
        break
    tp.append(sorted_neigborsSHINGLES[i][0])
    print "Shingles of Document " + str(sorted_neigborsSHINGLES[i][0]) + " with Jaccard Similarity " + str(
        round(sorted_neigborsSHINGLES[i][1], 2)) + "%"

    # Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)

print 'These are the True Positives, since no time saving assumptions were made while calculating the Jaccard similarity of shingles'
print "\nCalculating all Jaccard Similarities of Shingles Took %.2fsec" % elapsed
print '\nNote: In this section, we directly calculated the Jaccard similarities by comparing the shingle sets. This is included here to show how much slower it is than the MinHash and LSH approach.'
print '\nMoreover, the similarities calculated above are the actual similarities of the documents, since there were no assumption made'

# shingleNo =  shingleNo + s0
# print 'number', shingleNo

# Delete the Jaccard Similarities, since it's a pretty big matrix.
# del JSim

# =============================================================================
#                  Compare ALl Signatures & Display Similar Document Pairs
# =============================================================================
print 'Number of signatures', len(signatures)
# Count the true positives and false positives.
tpsig = 0
fpsig = 0

t0 = time.time()

threshold = 0
print '\nNow we will calculate Jaccard Similarity between signatures'
print "Values shown are the estimated Jaccard similarity"

# For each of the document pairs...
# for i in range(1, numDocs-1):
i = docid
signature1 = signatures[i]

neighbors_of_given_documentSIGNATURES = {}

for j in range(0, numDocs):
    if (i != j):
        signature2 = signatures[j]
        count = 0
        # Count the number of positions in the minhash signature which are equal.
        for k in range(0, numHashes):

            if (signature1[k] == signature2[k]):
                count = count + 1

        # Record the percentage of positions which matched.
        estJSim[getTriangleIndex(i, j)] = (count / float(numHashes))

        # Retrieve the estimated similarity value for this pair.
        # estJ = float(estJSim[getTriangleIndex(i, j)])

        # If the similarity is above the threshold...
        if float(estJSim[getTriangleIndex(i, j)]) > 0:

            # Calculate the actual Jaccard similarity for validation.
            s1 = set(signature1)
            s2 = set(signature2)

            J = len(s1.intersection(s2)) / float(len(s1.union(s2)))
            neighbors1 = []
            if (float(J) > threshold):
                percsimilarity = estJSim[getTriangleIndex(i, j)] * 100

                percJ = J * 100
                # Print out the match and similarity values with pretty spacing.
                # print "  %5s --> %5s   %.2f%s " % (docNames[i], docNames[j], percJ, '%')
                neighbors_of_given_documentSIGNATURES[j] = percJ

sorted_neigborsSIGNATURES = sorted(neighbors_of_given_documentSIGNATURES.items(), key=lambda x: x[1], reverse=True)
# print "Sorted Neighbors Signatures", sorted_neigbors, "%"

sigpos = []
print 'Comparing Signatures...'
print "The " + str(neighbors) + " closest neighbors of document " + str(docNames[i]) + " are:"
for i in range(0, neighbors):
    if i >= len(sorted_neigborsSIGNATURES):
        break
    print "Signatures of Document " + str(sorted_neigborsSIGNATURES[i][0]) + " with Jaccard Similarity " + str(
        round(sorted_neigborsSIGNATURES[i][1], 2)) + "%"
    sigpos.append(sorted_neigborsSIGNATURES[i][0])

fpsig = neighbors - len(list(set(tp).intersection(sigpos)))
tpsig = neighbors - fpsig
elapsed = (time.time() - t0)
print '\n', tpsig, '/', neighbors, 'True Positives and', fpsig, '/', neighbors, 'False Positives Produced While Comparing Signatures',

print "\nCalculating Jaccard Similarity of Signatures took %.2fsec" % elapsed

# apotelesma programmatos
# tupwnei to keimeno gia to opoio ginetai h anazhthsh omoiothtas
# ta x pio opoia me auto keimena, me to body tous, to id tous, kai to Jaccard similarity poy upologisthke

# lsh example https://www.youtube.com/watch?v=Arni-zkqMBA
# repeat procedure for lets say 20 times with 20 different hash tables and look at the buckets of all 20 cases.
# When 2 documents fall into the same bucket for at least 1 time it means that they might be similar
# every repetition, every different hash table results different buckets
# each time we should define the number of hyperplanes (separation lines) as well as the number of repetitions

# lsh computational cost
# n points, d-dimensional (=avg document length), k hyperplanes cutting through the space (k bit code)
# DK operations to find the right bucket for document A
# because in order to figure out in which bucket A will land we need
# to take A and do a dot product between A and each one of the normal vectors for the k hyperplanes
# once a lands in a bucket, we will compare A with every other document in that bucket
# average number of docs in a bucket
# if k hyperplanes (or bits) then we have 2^k possible hash codes or regions in space
# (number of possible intersections of hash spaces we can possibly have)
# N/(2^k) average points in a bucket = expected number of collisions in the bucket in a single hash table
# cost of comparisons DN/(2^k)
# dividing space into an exponential number of buckets and i have to repeat that L (# tables) times
# LSH = LDK + LDN/(2^K) -> O(logN)
# suppose that we set K to be logarithmic in terms of N (logN)
# if we make an index then the complexity is D(ND)/sqrt(ND) -> O(sqrt(N))
# brute force DN -> O(N)


# =============================================================================
#                                   LSH
# =============================================================================
# need to run these
# sudo apt-get install python3-pip
# sudo python3 -m pip install sortedcontainers
# pip install --upgrade pip
# pip install sortedcontainers
# easy_install sortedcontainers

while True:
    try:
        band_size = int(
            raw_input("\nPlease enter the size of the band. Valid band rows are 1 - " + str(numHashes) + ": "))
    except ValueError:
        print("Your input is not valid.")
        continue
    if band_size <= 0 or band_size > numHashes:
        print ("Your input is out of the defined range...")
        continue
    else:
        break

t0 = time.time()

tlist = []
for key, value in t.iteritems():
    temp = value
    tlist.append(temp)
# print tlist

# https://github.com/anthonygarvan/MinHash
from random import randint, seed, choice, random
import string
import sys
import itertools


def get_band_hashes(minhash_row, band_size):
    band_hashes = []
    for i in range(len(minhash_row)):
        if i % band_size == 0:
            if i > 0:
                band_hashes.append(band_hash)
            band_hash = 0
        band_hash += hash(minhash_row[i])
    return band_hashes


neighbors_of_given_documentLSH = {}


def get_similar_docs(docs, shingles, threshold, n_hashes, band_size, collectIndexes=True):
    t0 = time.time()
    lshsignatures = {}
    hash_bands = {}
    random_strings = [str(random()) for _ in range(n_hashes)]
    docNum = 0

    # for key, value in t.iteritems():
    #    temp = [key, doc]
    #   tlist.append(temp)
    w = 0
    # for doc in docs.iteritems():
    for doc in docs:

        lshsignatures[w] = doc
        # shingles = generate_shingles(doc, shingle_size)
        # print 'doc', doc
        # shingles = doc

        minhash_row = doc
        # print 'minhash_row', minhash_row, type(minhash_row)
        band_hashes = get_band_hashes(minhash_row, band_size)
        # print 'band_hashes', band_hashes
        w = w + 1
        docMember = docNum if collectIndexes else doc
        for i in range(len(band_hashes)):
            if i not in hash_bands:
                hash_bands[i] = {}
            if band_hashes[i] not in hash_bands[i]:
                hash_bands[i][band_hashes[i]] = [docMember]
            else:
                hash_bands[i][band_hashes[i]].append(docMember)
        docNum += 1

    similar_docs = set()
    similarity1 = []
    noPairs = 0
    print 'Comparing Signatures Found in the Same Buckets During LSH ...'
    # print "\n    Jaccard similarity After LSH\n"
    # print "    Pairs          Similarity"
    samebucketLSH = []
    samebucketcnt = 0
    for i in hash_bands:
        for hash_num in hash_bands[i]:
            if len(hash_bands[i][hash_num]) > 1:
                for pair in itertools.combinations(hash_bands[i][hash_num], r=2):
                    if pair not in similar_docs:
                        similar_docs.add(pair)
                        if pair[0] == docid and pair[1] != docid:

                            s1 = set(lshsignatures[pair[0]])
                            s2 = set(lshsignatures[pair[1]])

                            sim = len(s1.intersection(s2)) / float(len(s1.union(s2)))
                            if (float(sim) > threshold):
                                percsim = sim * 100
                                # print  "  %5s --> %5s   %.2f%s" % (pair[0], pair[1], percsim,'%')
                                noPairs = noPairs + 1
                                # return similar texts

                                # print 'TEXT WITH ID: ', pair[0], '\n AND BODY: ', body[pair[0]], '\n IS ', sim*100, '% SIMILAR TO', '\n TEXT WITH ID: ', pair[1], '\n AND BODY: ', body[pair[1]], '\n'
                            else:
                                percsim = 0
                            neighbors_of_given_documentLSH[pair[1]] = percsim
                            samebucketLSH.append(pair[1])
                            samebucketcnt = samebucketcnt + 1
                            elapsed = (time.time() - t0)

    print 'Number of false positives while comparing signatures which were found in the same bucket',
    sorted_neigborsLSH = sorted(neighbors_of_given_documentLSH.items(), key=lambda x: x[1], reverse=True)
    # print "Sorted Neighbors Signatures", sorted_neigbors, "%"

    lshpos = []
    print 'Comparing Signatures Found in the Same Bucket During LSH...'
    print "The " + str(neighbors) + " closest neighbors of document " + str(docid) + " are:"
    for i in range(0, neighbors):
        if i >= len(sorted_neigborsLSH):
            break
        if sorted_neigborsLSH[i][1] > 0:
            print "\nChosen Signatures (After LSH) of Document " + str(sorted_neigborsLSH[i][0]) + " with Jaccard Similarity " + str(round(sorted_neigborsLSH[i][1], 2)) + "%"
            print "\nBody of document " + str(sorted_neigborsLSH[i][0]) + "\n" + str(printedbodies[sorted_neigborsLSH[i][0]])
            lshpos.append(sorted_neigborsLSH[i][0])

    neighborsfplsh = neighbors - len(list(set(tp).intersection(lshpos)))
    neighborstplsh = neighbors - neighborsfplsh
    # totalfplsh =
    totaltplsh = len(list(set(tp).intersection(samebucketLSH)))
    totalfplsh = samebucketcnt - totaltplsh

    print '\nEvaluating the', neighbors, 'neighbors produced by LSH...'
    print neighborstplsh, 'out of', neighbors, 'TP and', neighborsfplsh, 'out of', neighbors, 'FP'
    print '\nEvaluating the', samebucketcnt, 'pairs which fell in the same bucket...'

    if samebucketcnt > 0:
        prctpLSH = round((totaltplsh / float(samebucketcnt)) * 100, 2)
        prcfpLSH = 100 - prctpLSH
        print totaltplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are TP', prctpLSH, '%'
        print totalfplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are FP', prcfpLSH, '%'
    else:
        print totaltplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are TP'
        print totalfplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are FP'

    return similar_docs


# Report how long shingling took.

n_hashes = numHashes

n_similar_docs = 2
seed(42)

finalshingles = docsAsShingleSets
# print 'docs', docs, type(docs)
# print tlist
# print docs, type(docs)
# print signatures, type(signatures)


similar_docs = get_similar_docs(signatures, finalshingles, threshold, n_hashes, band_size, collectIndexes=True)

print '\nLocality Sensitive Hashing ' + str(len(signatures)) + ' docs took %.2f sec.' % (time.time() - t0)


r = float(n_hashes / band_size)
similarity = (1 / r) ** (1 / float(band_size))
