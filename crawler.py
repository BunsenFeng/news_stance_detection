import urllib.request
url = "https://www.nytimes.com/2021/07/12/us/politics/pfizer-booster-shots.html"
file = urllib.request.urlopen(url)

for line in file:
	decoded_line = line.decode("utf-8")
	print(decoded_line)