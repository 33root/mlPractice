#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
	"""
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
	"""

	cleaned_data = []
	data = []
    ### your code goes here
	data = []
	for i, val in enumerate(predictions):
		error = predictions[i][0]- net_worths[i][0]
		tup = (ages[i][0],net_worths[i][0],error)
		data.append(tup)
	
	data.sort(key=lambda tup: tup[2])
	elements_to_keep = len(data)* 0.9	

	cleaned_data = data[:int(elements_to_keep)]

	return cleaned_data

