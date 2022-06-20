#使用拉格朗日乘子法
from sympy import *

x1 = Symbol("x1")
x2 = Symbol("x2")
a = Symbol("a")
b = Symbol("b")
f = 10 - x1**2 - x2**2 +a*(x1 + x2) + b*(x1**2 - x2)
#f对x1求导数
fx1 = diff(f,x1)
#f对x2求导数
fx2 = diff(f,x2)
#fx1=0，fx2=0，
result = solve([fx1,fx2,(x1**2-x2)*b,x1+x2],[x1,x2,a,b])

for i in range(len(result)):
    if result[i][3]>=0 and result[i][0]**2-result[i][1]<=0 and result[i][2]!=0:
        print(result[i])
        print("loss:",10 - result[i][0]**2 - result[i][1]**2 +result[i][2]*(result[i][1] + result[i][0]) + result[i][3]*(result[i][0]**2 - result[i][1]))