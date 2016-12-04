# python script to load colors from a csv file assuming first entries are strings and then floats
def parseStringLine(line_str, num_floats, num_str) :

	acceptble_delim = [',', '\n']

	size = len(line_str)
	if line_str[size-1] != '\n' :
		line_str += '\n'

	num_text = ''
	val_array = []
	val_count = 0
	str_count = 0

	flag_read_strings = False

	for c in line_str :

		if flag_read_strings == False :
			if c in acceptble_delim :
				val = num_text
				val_array.append(val)
				str_count += 1
				flag_read_strings = True
				# print num_text
				# print val
				num_text = ''
			elif c.isalpha():
				num_text += c

		else :
			if c in acceptble_delim :
				try:
					val = float(num_text)
				except ValueError:
					print "ERROR : One of the numbers in csv file was invalid"
					return None
				val_count += 1
				val_array.append(val)
				# print num_text
				# print val
				num_text = ''
			elif c.isdigit() or c == '.' or c=='-' or c=='e':
				num_text += c

	if val_count == num_floats and str_count == num_str:
		return val_array
	else :
		print "ERROR : Expecting " + str(num_floats) + "  numbers in a single line in csv file. Got " + str(val_count) 
		return None

def parseCSVMatrix(file_name, num_rows) :

	fp = open(file_name, "r")

	line_count = 0
	mat = []
	
	while True :
		
		string_line = fp.readline()

		if string_line is None or len(string_line) == 0:
			break

		arr = parseStringLine(string_line, 3, 1)

		if arr != None :
			line_count += 1
			# print arr
			mat.append(arr)	
		else :
			print "ERROR : Parsing of " + file_name + " failed"
			return None

	if line_count == num_rows :
		return mat
	else :
		print "ERROR : Parsing of " + file_name + " failed"
		print "Expecting " + str(num_rows) + "rows in csv file. Got " + str(line_count)
		return None

	fp.close()
