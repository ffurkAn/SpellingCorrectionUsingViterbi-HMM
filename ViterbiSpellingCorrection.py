import csv
from collections import Counter

import numpy as np

# kabul
# 20000 karakter okuduktan sonra  okunmaya başlanan ilk kelime parçaso

# CONSTANTS
numberOfTestCharacter = 20000
numberOfWords = 0
delimiterConstant = ' '

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

# B[x][o] : (Emission Probability Matrix) Çıkış Olasılık Matrisi: Eğitim setinde x harfi olması gerekirken o harfinin görülme olasılığı.
countOfEmissionProbabilityMatrix = np.zeros(shape=(26, 27))


################################################################### FUNCTIONS DEFINITIONS #############################################################################


# ilk durum harf olasılıklarının çıkartılması
def createFirstStateLetterPossibilitiesVector( ):
	for letterIndex in range(0, 26):
		# her harf 1 ile ilklendiği için 1 çıkarıyoruz
		numberOfLetterInTrainSet = letterCounts[alphabet[letterIndex]] - 1
		probabilityOfWordStartsWithLetter.append(numberOfLetterInTrainSet / numberOfWords)
	return


# emission ve transition matrix için olasılık matrisi
def createProbabilityMatrixOfCountMatrix( countMatrix ):
	probabilityMatrix = np.zeros(shape=(26, 26))
	for row in range(0, 26):
		total = countMatrix[row][26]
		if (total == 0):
			continue
		else:
			for column in range(0, 26):
				count = countMatrix[row][column]
				probability = count / total
				probabilityMatrix[row][column] = probability
	return probabilityMatrix


# hangi harften sonra hangi harf deliyor sayısını tutmak için
def incrementCountOfTransitionProbabilityMatrix( previousLetter, currentLetter ):
	previousLetterIndex = alphabetEnum[previousLetter]
	currentLetterIndex = alphabetEnum[currentLetter]
	countOfTransitionProbabilityMatrix[previousLetterIndex, currentLetterIndex] += 1
	countOfTransitionProbabilityMatrix[previousLetterIndex, 26] += 1  # totalı arttır
	return


def incrementCountOfEmissionProbabilityMatrix( mustBe, observed ):
	mustBeLetterIndex = alphabetEnum[mustBe]
	observerdLetterIndex = alphabetEnum[observed]
	countOfEmissionProbabilityMatrix[mustBeLetterIndex, observerdLetterIndex] += 1
	countOfEmissionProbabilityMatrix[mustBeLetterIndex, 26] += 1
	return


# viterbi algorithm
def runViterbi( testWord, startProb, transition, emission ):
	sigmaMatrix = np.zeros(shape=(len(testWord), len(alphabet)))

	# test kelimesinin ilk harfi için ilk durum olasılıkları hesaplanıyor
	for charIndex in range(0, len(alphabet)):
		sigmaMatrix[0][charIndex] = startProb[charIndex] * emission[charIndex][alphabetEnum[testWord[0]]]

	# test kelimesinin geri kalan harfleri için
	for i in range(1, len(testWord)):
		for state in range(0, len(alphabet)):
			maxProb = 0
			for prevState in range(0, len(alphabet)):
				prob = emission[alphabetEnum[testWord[i]]][state] * transition[prevState][state] * sigmaMatrix[i - 1][
					prevState]
				if (prob > maxProb):
					maxProb = prob
			sigmaMatrix[i][state] = maxProb

	predictedWord = ""
	for i in range(0, testWord.__len__()):
		predictedChar = alphabet[np.argmax(sigmaMatrix[i])]
		predictedWord += predictedChar

	return predictedWord

######################################################################## BEGINNING OF PROGRAM ###################################################################

# reads file as numberOfline X 2 matrix
f = open('docs.data')
reader = csv.reader(f, delimiter=delimiterConstant)

# dosyadaki egitimdeki harflerin sayıları ve transition matrix sayıları hesaplanır
testCharacterCounter = 0
totalCharacterCounter = 0

# calculate the size of test matrix
for row in reader:
	if (testCharacterCounter < numberOfTestCharacter):
		totalCharacterCounter += 1
		if row[0] != "_":
			testCharacterCounter +=1

# create test and train matrix
testMatrix = np.empty(shape=(totalCharacterCounter, 2), dtype=np.str)
numberOfTrainCharacters = 0
previousLetter = ""
counter = 0

f = open('docs.data')
reader = csv.reader(f, delimiter=delimiterConstant)

for row in reader:
	# test section
	if (counter < totalCharacterCounter):
		testMatrix[counter] = row
		counter += 1

	# train section
	else:
		leftSideLetter = row[0]  # true
		rightSideLetter = row[1]  # false

		# ilk egitim datasında önceki harf yok sonraki harften devam et
		if (previousLetter == ""):
			previousLetter = leftSideLetter
			continue

		if (leftSideLetter != "_"):
			# dosyanın ilk karakteri ya da önceki karakter underscore ise yeni kelime sayısını 1 arttır
			if (previousLetter == "_"):
				numberOfWords += 1
				letterCounts[leftSideLetter] += 1
				previousLetter = leftSideLetter

			# ilk karakter ve önceki karakter underscore ise matrix update edilmez
			else:
				incrementCountOfTransitionProbabilityMatrix(previousLetter, leftSideLetter)
				# eğer soldaki ile sağdaki aynı karakter değilse emission matrixi update et
				if (leftSideLetter != rightSideLetter):
					incrementCountOfEmissionProbabilityMatrix(leftSideLetter, rightSideLetter)

				previousLetter = leftSideLetter
		else:
			previousLetter = "_"

# ilk durum harf olasılıklarının çıkartılması
createFirstStateLetterPossibilitiesVector()

# transition matrix olasılıklarının çıkarılması
transitionProbabilityMatrix = createProbabilityMatrixOfCountMatrix(countOfTransitionProbabilityMatrix)

# emission matrix olasılıklarının çıkarılması
emissionProbabilityMatrix = createProbabilityMatrixOfCountMatrix(countOfEmissionProbabilityMatrix)

testWord = ""
trueWord = ""
numberOfTrueCorrection = 0
numberOfFalseCorrection = 0

for row in testMatrix:
	# use the right side of doc
	if (row[1] == "_"):
		print("------------------------------")
		print("Test:         " + testWord)
		expectedWord = runViterbi(testWord, probabilityOfWordStartsWithLetter, transitionProbabilityMatrix,
		                          emissionProbabilityMatrix)
		print("Predicted:    " + expectedWord)
		print("Right one is: " + trueWord)

		if (testWord == expectedWord):
			numberOfTrueCorrection += 1
		else:
			numberOfFalseCorrection += 1

		testWord = ""
		trueWord = ""
	else:
		testWord += row[1]
		trueWord += row[0]

accuracy = (100 * numberOfTrueCorrection) / (numberOfFalseCorrection + numberOfTrueCorrection)
print("Accuracy: " + accuracy.__str__())

a = [[0.5, 0.4, 0.1],
     [0.4, 0.3, 0.3],
     [0.1, 0.4, 0.5]]

b = [[0.5, 0.75, 0.25],
     [0.5, 0.25, 0.75]]

pi = [0.5, 0.3, 0.2]

o = "srrsr"
