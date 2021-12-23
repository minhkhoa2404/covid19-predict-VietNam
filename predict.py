from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

req = requests.get(
    'https://tuoitre.io/covid/bieu-do')
soup = BeautifulSoup(req.text, 'html.parser')

# Find all number that are inside square brackets []
lst = re.findall('\[(.*?)\]', str(soup))

covidData = []
xLabel = []

# Modify datalist
for i in lst:
    if i.find('[') != -1:
        lst.insert(lst.index(i), i.replace('[', ''))
        lst.remove(i)

# Split , in list to form a 2d array
for i in lst:
    lst.insert(lst.index(i), i.split(','))
    lst.remove(i)
j = 1
# Extract data in list
for i in lst:
    covidData.append(i[1])
    xLabel.append(j)
    j += 1

modifyArray = covidData[:]
modifyArray = np.array(modifyArray, dtype=int)
covidData = list(map(int, covidData))
highestAllowed = np.mean(modifyArray) + 3*np.std(modifyArray)
lowestAllowed = np.mean(modifyArray) - 3*np.std(modifyArray)
for i in covidData:
    if i > highestAllowed or i < lowestAllowed or i == 0:
        ind = covidData.index(i)
        covidData.remove(i)
        xLabel.remove(xLabel[ind])

covidData = np.array(covidData, dtype=int)
xLabel = np.array(xLabel, dtype=int).reshape(-1, 1)
poly_reg = PolynomialFeatures(degree=9)
x_poly = poly_reg.fit_transform(xLabel)

model = LinearRegression()
model.fit(x_poly, covidData)

pred = model.predict(poly_reg.fit_transform(xLabel))

Rsquare2 = r2_score(covidData, pred)
print(f'Score: {Rsquare2}')

print(model.predict(poly_reg.fit_transform(
    np.array([241, 242, 243, 244, 245, 246, 247, 248, 249, 250]).reshape(-1, 1))))
plt.figure(figsize=(20, 5))
plt.scatter(xLabel, covidData, color='black')
plt.plot(covidData, color='red', label='Real')
plt.plot(pred, color='blue', label='Predict')
plt.title('Covid-19 cases')
plt.legend()
plt.show()
