import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_prediction(i, y, input_size, future):
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(input_size), yi[:input_size], color, linewidth = 2.0)
        plt.plot(np.arange(input_size, input_size + future), yi[input_size:], color + ':', linewidth = 2.0)
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.savefig('predict%d.pdf'%i)
    plt.close()

def draw_convergence(train_losses, test_losses, recorded_steps, depthH, breadth, lr, steps):
    plt.figure()
    plt.title('Convergence (train=%d, test=%d)' % (train_losses[-1], test_losses[-1]))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.plot(recorded_steps, train_losses, 'b')
    plt.plot(recorded_steps, test_losses, 'r')
    plt.yscale('log')
    plt.legend(['Training', 'Testing'])
    plt.savefig('convergence_%d_%dx%d_%.2flr_%ds.pdf' % (test_losses[-1], depthH, breadth, lr, steps))
