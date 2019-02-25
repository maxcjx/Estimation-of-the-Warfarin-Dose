import sys
import csv
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# row[1]: Gender
# row[2]: Race
# row[4]: Age
# row[5]: Height
# row[6]: Weight
# row[23]: Amiodarone
# row[24]: Carbamazepine
# row[25]: Phenytoin (Dilantin)
# row[26]: Rifampin or Rifampicin
# row[34]: Therapeutic Dose of Warfarin

def isComplete(row):
	if row[34] == 'NA' or row[4] == 'NA' or row[5] == 'NA' or row[6] == 'NA' or \
	   is_number(row[34]) == False or is_number(row[5]) == False or is_number(row[6]) == False:
	   return False
	return True


def ageBucket(age):
	end = 0
	for i, c in enumerate(age):
		if c.isdigit() == False:
			end = i
			break

	return int(age[0:end])/10


def dosageBucket(weeklyDosage):
	dailyDosage = weeklyDosage*1.0/7
	if dailyDosage < 3:
		return 0
	elif dailyDosage <= 7:
		return 1
	else:
		return 2

def compute():

	total = 0
	fixedDose_cnt = 0
	WCDA_cnt = 0
	infile = "./data/warfarin.csv"
	with open(infile) as f:
		reader = csv.reader(f)
		firstLine = True

		for row in reader:
			if firstLine == True:
				firstLine = False
			else:

				if isComplete(row) == False:
					continue

				trueDose = float(row[34])

				fixedDose = 5*7
				if dosageBucket(trueDose) == dosageBucket(fixedDose):
					fixedDose_cnt += 1

				WCDA = 4.0376
				WCDA -= 0.2546 * ageBucket(row[4])
				WCDA += 0.0118 * float(row[5])
				WCDA += 0.0134 * float(row[6])

				if row[2] == 'Asian':
					WCDA -= 0.6752
				elif row[3] == 'Black or African American':
					WCDA += 0.4060
				else:
					WCDA += 0.0443

				if row[24] == '1' or row[25] == '1' or row[26] == '1':
					WCDA += 1.2799

				if row[23] == '1':
					WCDA -= 0.5695

				if dosageBucket(trueDose) == dosageBucket(WCDA*WCDA):
					WCDA_cnt += 1

				total += 1

	print("Accuracy for Fixed-dose Algorithm is : ", fixedDose_cnt*1.0/total)
	print("Accuracy for Warfarin Clinical Dosing Algorithm is : ", WCDA_cnt*1.0/total)

if __name__ == '__main__':
    if len(sys.argv) != 1:
    	raise Exception("usage: python baseline.py")
    compute()
