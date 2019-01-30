import numpy as np

def distance(encoding1, encoding2):
	"""
	Calculate the euclidean distance between 2 encoding, each encoding is a 128x1 vectr.
	""" 
	encoding1 = np.array(encoding1)
	encoding2 = np.array(encoding2)

	return np.linalg.norm(encoding2 - encoding1)

# def compare():