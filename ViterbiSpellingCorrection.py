from collections import Counter

import numpy as np

# CONSTANTS
numberOfTrainLetter = 20000

# her harf sayacı 1 ile ilkleniyor. daha sonra 1 çıkartılarak initialize edilecek
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# holds the number of letters
letterCounts = Counter(alphabet)

## alphabet dictionary: enumerate etmek için kullanılacak
alphabetEnum = {
	'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13,
	'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25
	}

firstStateLetterPossibilities = []  # FIRST STATE VECTOR

# harflerden sonra gelen harflerin sayıları tutulucak.daha sonra oranların tutulacağı matrix oluşturulacak
# son kolon da o harten sonra gelen harflerin toplamları yer alıcak ki %delik dilimi kolay hesaplayabilelim
countOfStateTransitionMatrix = np.zeros(shape=(26, 27))
stateTransitionMatrix = np.zeros(shape=(26, 26))

f = open('docs.data')


# hangi harften sonra hangi harf deliyor sayısını tutmak için
def incrementCountOfStateTransitionMatrix( previousLetter, currentLetter ):
	previousLetterIndex = alphabetEnum[previousLetter]
	currentLetterIndex = alphabetEnum[currentLetter]
	countOfStateTransitionMatrix[previousLetterIndex, currentLetterIndex] += 1
	countOfStateTransitionMatrix[previousLetterIndex, 26] += 1  # totalı arttır
	return

# ilk durum harf olasılıklarının çıkartılması
def createFirstStateLetterPossibilitiesVector( ):
	for letterIndex in range(0, 26):
		numberOfLetterInTrainSet = letterCounts[
			                           alphabet[letterIndex]] - 1  # her harf 1 ile ilklendiği için 1 çıkarıyoruz
		firstStateLetterPossibilities.append((100 * numberOfLetterInTrainSet) / (numberOfTrainLetter))
	return


# transition matrix olasılıklarının çıkarılması
def createStateTransitionMatrix( ):
	for row in range(0, 26):
		total = countOfStateTransitionMatrix[row][26]
		for column in range(0, 26):
			count = countOfStateTransitionMatrix[row][column]
			possibility = (100 * count) / total
			stateTransitionMatrix[row][column] = possibility
	return


# dosyadaki egitimdeki harflerin sayıları ve transition matrix sayıları hesaplanır
trainCounter = 0
previousLetter = ""
for line in f.readlines():
	if (trainCounter > numberOfTrainLetter):
		break

	currentLetter = line.split()[0]
	if (currentLetter == ' ') or (currentLetter == "_"):
		continue
	else:
		letterCounts[currentLetter] += 1
		trainCounter += 1
		if (trainCounter > 1):
			incrementCountOfStateTransitionMatrix(previousLetter, currentLetter)
		previousLetter = currentLetter

# ilk durum harf olasılıklarının çıkartılması
createFirstStateLetterPossibilitiesVector()

# transition matrix olasılıklarının çıkarılması
createStateTransitionMatrix()

f.close()
