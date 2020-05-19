import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08
    plt.scatter(x,y, color='red', marker='+', linewidths='5')
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        if(i==999):
            plt.plot(x,y_predicted, color='green')
        else:    
            plt.plot(x,y_predicted, color='blue')
        cost = (1/n)*sum(val**2 for val in (y-y_predicted))
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - md*learning_rate
        b_curr = b_curr - bd*learning_rate
        print(f'm {m_curr}, b {b_curr}, cost {cost}, iteration {i}')
    plt.show()

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)