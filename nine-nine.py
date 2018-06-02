for x in range(0,9):
	for y in range(0,9):
		print('{0}'.format('%2d ' % ((x+1) * (y+1))), end="")
	print('')