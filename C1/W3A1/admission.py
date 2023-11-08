import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy


a = 0.5

df = pd.read_csv("ex2data1.txt", header=None)

w = np.array([1, 1])
x = list()
xmax = 100

for row in df.iterrows():
	x1 = row[1][0]/xmax
	x2 = row[1][1]/xmax
	xarr = [x1, x2]
	x.append(xarr)

x = np.array([np.array(xi) for xi in x])

y = df[2].to_numpy()
b = 0


def getz(w, b, xi):
	z = np.dot(w, xi) + b
	return z

def model(w, b, xi):
	z = getz(w, b, xi)
	f = 1/(1 + (math.e**(-z)))
	return f

def cost_func(w, b):
	cost = 0	
	for i in range(0, len(x)):
		yhat = model(w, b, x[i])
		yi = y[i]
		costpart = ((-1)*(yi)*(math.log(yhat))) - ((1-yi)*(math.log(1-yhat)))
		cost = cost + costpart
	cost = cost/len(x)
	return cost
		
def grad_desc():
	global w
	global b
	d = [0,0,0]
	for i in range(0, len(x)):
		#print("descending")
		yhat = model(w, b, x[i])
		#print(yhat)
		#print(y[i])
		d[0] = d[0] + ((yhat - y[i])*(x[i][0]))
		d[1] = d[1] + ((yhat - y[i])*(x[i][1]))
		d[2] = d[2] + (yhat - y[i])
	d = np.array(d)
	d = d/len(x)
	w = w - (a*d[0:2])
	b = b - (a*d[2])
	print(d)
	print(w)
	print(b)


costs = list()
costs.append(cost_func(w,b))
print(costs[0])
while len(costs)<5000:
	prevw = copy.deepcopy(w)
	prevb = copy.deepcopy(b)
	grad_desc()
	cost = cost_func(w, b)
	costs.append(cost)
	print(cost)
	if cost == costs[-2]:
		print("No change in cost")
		break
	if abs(cost) > abs(costs[-2]):
		print("Cost increased")
		w = prevw
		b = prevb
		break

	

print(w)
print(b)


plt.plot(np.arange(0, len(costs)), costs)
plt.show()


yhatlist = list()
for i in range(0, len(y)):	
	yhatlist.append(model(w, b, x[i]))

fig, al = plt.subplots()
al.scatter(df[df[2]==1][0], df[df[2]==1][1], color = 'blue', marker='x')
al.scatter(df[df[2]==0][0], df[df[2]==0][1], color = 'red', marker='x')
al.plot([(-b/w[0])*xmax, 0], [0, (-b/w[1])*xmax], color = 'black')
al.set_xlim(0, xmax)
al.set_ylim(0, xmax)
plt.show()


while True:
	print()
	s1 = input("Score 1: ")
	if s1=="":
		break
	else:
		s1 = int(s1)
	s2 = input("Score 2: ")
	if s2 == "":
		continue
	else:
		s2 = int(s2)

	f = model(w, b, np.array([s1/xmax, s2/xmax]))
	print(f)
	if f>0.5:
		print("Yes")
	elif f<0.5:
		print("No")
	else:
		print("?")
