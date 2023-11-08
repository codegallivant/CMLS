import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
import copy

a = 0.1
# a = 0.01


df = pd.read_csv("ex1data1.txt", header = None)
print(df)

#Cities population in 10,000s
xdata = df[0].to_numpy()
xmax = np.max(xdata)
# xmax = 1
xdata = xdata/xmax
#Average monthly profits
ydata = df[1].to_numpy()
ymax = np.max(ydata)
# ymax = 1
ydata = ydata/ymax

print(xdata)
print(ydata)

m = len(xdata)


#Visualise data - 
# fig, ax = plt.subplots()
# ax.scatter(xdata, ydata, marker='x')
# plt.xlabel("Average monthly profits")
# plt.ylabel("Population in 10,000s")
# plt.title("Restaurant Stats")
# plt.show()

def compute_model(w, b, x):
	 return w*x + b

def cost_func(w, b):
	J=0
	for i in range(1, m+1):
		J = J + (compute_model(w, b, xdata[i-1]) - ydata[i-1])**2
	J = J * (1/2) * (1/m)
	return J

def randwb():
	w = 1
	b = 0
	return [w, b]


w, b = randwb()
initialcost = cost_func(w, b)
print(w,b)
print(initialcost)
costs = list()
costs.append(initialcost)
while costs[-1] > 0.0000001:
	dw = 0
	db = 0
	for i in range(1, m+1):
		dw = dw + (compute_model(w, b, xdata[i-1]) - ydata[i-1])*(xdata[i-1])
		db = db + (compute_model(w, b, xdata[i-1]) - ydata[i-1])
	dw = dw/m
	db = db/m
	tmp_w = w - (a*dw)
	tmp_b = b - (a*db) 
	w = copy.deepcopy(tmp_w)
	b = copy.deepcopy(tmp_b)
	nowcost = cost_func(w, b)
	print(nowcost)
	prevcost = costs[-1]
	costs.append(nowcost)
	if len(costs)>=2:
		if nowcost > prevcost:
			print("Cost increased")
			break
		elif nowcost == prevcost:
			print("no change in cost")
			break



plt.plot(np.arange(1, len(costs)+1), costs)
plt.show()


fig, al = plt.subplots()


# wx+b
# # (0, b), (-b/w, 0)
print()
print(w)
print(b)
# al.plot([0, -b/w], [b, 0])
xdata = xdata * xmax
ydata = ydata * ymax
w = w * (ymax/xmax)
b = b * ymax
print(w)
print(b)
print(cost_func(w,b))
al.scatter(xdata, ydata, marker='x', color = 'red')
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(al.get_xlim())
    y_vals = intercept + slope * x_vals
    al.plot(x_vals, y_vals)
abline(w, b)
al.set_xlabel("Population in 10,000s")
al.set_ylabel("Average monthly profits in $10,000s")
al.set_title("Restaurant Stats and Prediction")
plt.show()


def predict(x):
	return ((w*(x/10000)) + b)*10000


while True:
	print()
	p = input("Population? ")
	if p == "":
		break
	else:
		p = float(p)
		print("Monthly profit (predicted estimate): ",predict(p))
