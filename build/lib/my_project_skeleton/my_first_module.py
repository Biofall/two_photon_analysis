import numpy as np
import matplotlib.pyplot as plt

def plot():
	"""
	This is an example plot that uses numpy and
	matplotlib.pyplot to imshow a random data.
	"""
	random_dat = np.random.random([30,30])
	fig, ax = plt.subplots()
	ax.imshow(random_dat)
	ax.set_title("HELLOOO")

	return fig, ax

if __name__ == "__main__":
	print("yooo, running as a module")
	plot()
	plt.show()
