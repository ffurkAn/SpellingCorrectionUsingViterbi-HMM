import numpy as np

alphabet = ['h', 'm', 'l']
alphabetEnum = {'h': 0, 'm': 1, 'l': 2}

states = ['s', 'r']
statesEnum = {'s': 0, 'r': 1}


# viterbi algorithm
def runViterbi( testWord, startProb, transition, emission ):
	sigmaMatrix = np.zeros(shape=(len(testWord), len(alphabet)))

	# test kelimesinin ilk harfi için ilk durum olasılıkları hesaplanıyor
	for charIndex in range(0, len(alphabet)):
		sigmaMatrix[0][charIndex] = startProb[charIndex] * emission[charIndex][statesEnum[testWord[0]]]

	# test kelimesinin geri kalan harfleri için
	for i in range(1, len(testWord)):
		for state in range(0, len(alphabet)):
			maxProb = 0
			for prevState in range(0, len(alphabet)):
				prob = emission[state][statesEnum[testWord[i]]] * transition[prevState][state] * sigmaMatrix[i - 1][
					prevState]
				if (prob > maxProb):
					maxProb = prob
			sigmaMatrix[i][state] = maxProb

	predictedWord = ""
	for i in range(0, testWord.__len__()):
		predictedChar = alphabet[np.argmax(sigmaMatrix[i])]
		predictedWord += predictedChar

	return predictedWord


#   H M L
# H
# M
# L
a = [[0.5, 0.4, 0.1],
     [0.4, 0.3, 0.3],
     [0.1, 0.4, 0.5]]

#   s r
# H
# M
# L
b = [[0.75, 0.25],
     [0.5, 0.5],
     [0.25, 0.75]]

pi = [0.2, 0.5, 0.3]

o = "srrsr"

predictedHidden = runViterbi(o, pi, a, b)

print(predictedHidden)
