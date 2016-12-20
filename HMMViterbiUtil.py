################################################################### FUNCTIONS DEFINITIONS #############################################################################


# ilk durum harf olasılıklarının çıkartılması
def createFirstStateLetterPossibilitiesVector( ):
	for letterIndex in range(0, 26):
		# her harf 1 ile ilklendiği için 1 çıkarıyoruz
		numberOfLetterInTrainSet = letterCounts[alphabet[letterIndex]] - 1
		probabilityOfWordStartsWithLetter.append((100 * numberOfLetterInTrainSet) / (numberOfWords))
	return


# emission ve transition matrix için olasılık matrisi
def createProbabilityMatrixOfCountMatrix( countMatrix, probMatrix ):
	for row in range(0, 26):
		total = countMatrix[row][26]
		if (total == 0):
			continue
		else:
			for column in range(0, 26):
				count = countMatrix[row][column]
				probability = (100 * count) / total
				probMatrix[row][column] = probability
	return


# emission ve transition matrixlerin adetlerini tutuyor
# transition matrix için x: previous, y: current letter temsil eder
# emission matrix için x: olması gereken harf, y ise gözlenen yanlış harfi temsil eder
def incrementCountOfMatrixByTarget( x, y, targetCountMatrix ):
	xIndex = alphabetEnum[x]
	yIndex = alphabetEnum[y]
	targetCountMatrix[xIndex, yIndex] += 1
	targetCountMatrix[xIndex, 26] += 1
	return
