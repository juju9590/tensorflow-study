x = 10
y = 10
w = 0.001
lr = 0.001
epochs = 2000

for i, e in enumerate(range(epochs)):
    hypothesis = x * w
    loss = (hypothesis - y) **2 # mse, bias 제외, 간략하게만

    print(i + 1, "Loss :", round(loss, 4), '\tPredict :', round(hypothesis, 4))

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if (up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr

# Results
# Loss : 0.0      Predict : 10.0
# Loss : 0.0001   Predict : 9.99
# Loss : 0.0      Predict : 10.0
# Loss : 0.0001   Predict : 9.99
# Loss : 0.0      Predict : 10.0