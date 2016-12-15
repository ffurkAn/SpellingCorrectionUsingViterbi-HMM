from collections import Counter

import numpy as np

# CONSTANTS
numberOfTrainLetter = 20000
numberOfWords = 0

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

probabilityOfWordStartsWithLetter = []  # FIRST STATE VECTOR

# A[x][y] : (Transtion Probability Matrix) Durum Geçiş Olasılık Matrisi: Eğitim setinde bir kelimede x harfinden sonra y harfi gelme olasılığı
countOfTransitionProbabilityMatrix = np.zeros(shape=(26, 27))
transitionProbabilityMatrix = np.zeros(shape=(26, 26))

# B[x][o] : (Emission Probability Matrix) Çıkış Olasılık Matrisi: Eğitim setinde x harfi olması gerekirken o harfinin görülme olasılığı.
countOfEmissionTransitionMatrix = np.zeros(shape=(26, 27))
emissionTransitionMatrix = np.zeros(shape=(26, 26))


################################################################### FUNCTIONS DEFINITIONS #############################################################################

# hangi harften sonra hangi harf deliyor sayısını tutmak için
def incrementCountOfStateTransitionMatrix( previousLetter, currentLetter ):
	previousLetterIndex = alphabetEnum[previousLetter]
	currentLetterIndex = alphabetEnum[currentLetter]
	countOfTransitionProbabilityMatrix[previousLetterIndex, currentLetterIndex] += 1
	countOfTransitionProbabilityMatrix[previousLetterIndex, 26] += 1  # totalı arttır
	return

# ilk durum harf olasılıklarının çıkartılması
def createFirstStateLetterPossibilitiesVector( ):
	for letterIndex in range(0, 26):
		# her harf 1 ile ilklendiği için 1 çıkarıyoruz
		numberOfLetterInTrainSet = letterCounts[alphabet[letterIndex]] - 1
		probabilityOfWordStartsWithLetter.append((100 * numberOfLetterInTrainSet) / (numberOfWords))
	return


# transition matrix olasılıklarının çıkarılması
def createStateTransitionMatrix( ):
	for row in range(0, 26):
		total = countOfTransitionProbabilityMatrix[row][26]
		if (total == 0):
			continue
		else:
			for column in range(0, 26):
				count = countOfTransitionProbabilityMatrix[row][column]
				probability = (100 * count) / total
				transitionProbabilityMatrix[row][column] = probability
	return


######################################################################## BEGINNING OF PROGRAM ###################################################################

f = open('docstest.data')

# dosyadaki egitimdeki harflerin sayıları ve transition matrix sayıları hesaplanır
trainCounter = 0
previousLetter = ""
for line in f.readlines():
	if (trainCounter > numberOfTrainLetter):
		break

	currentLetter = line.split()[0]
	if (currentLetter == "_"):
		previousLetter = currentLetter
	else:
		# dosyanın ilk karakteri ya da önceki karakter underscore ise yeni kelime sayısını 1 arttır
		if (previousLetter == "_") or (trainCounter == 0):
			numberOfWords += 1

		letterCounts[currentLetter] += 1
		trainCounter += 1
		# ilk karakter ve önceki karakter underscore ise matrix update edilmez
		if (trainCounter > 1) and (previousLetter != "_"):
			incrementCountOfStateTransitionMatrix(previousLetter, currentLetter)
		previousLetter = currentLetter

# ilk durum harf olasılıklarının çıkartılması
createFirstStateLetterPossibilitiesVector()

# transition matrix olasılıklarının çıkarılması
createStateTransitionMatrix()

f.close()
