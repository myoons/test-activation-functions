# Activation Functions Test

Test 6 activation functions for training a single node. (1 weight, 1 bias)

- Input : 1.
- Initial weight : -1.
- Initial bias : -1.
- Target : 1.

 
I have done this test because I was curious that since GeLU and swish function has negative derivative (in some part), training could be messed.
So I manipulated initial output (before activation function) to have negative derivatives and could see that it isn't trained.
Actually, it is trained to opposite side.
Below are images of Sigmoid, ReLU, GeLU. You can find more cases in images folder.

## Sigmoid
![sigmoid](https://user-images.githubusercontent.com/67945103/120077898-b2857080-c0e7-11eb-95f5-b403ed02ce4e.png)
![sigmoid_derivative](https://user-images.githubusercontent.com/67945103/120077899-b44f3400-c0e7-11eb-9a84-f4d08a4b1afc.png)
![sigmoid_result](https://user-images.githubusercontent.com/67945103/120077901-b5806100-c0e7-11eb-9030-78948c4fc7a5.png)


## ReLU (Dying)

![relu](https://user-images.githubusercontent.com/67945103/120077921-d052d580-c0e7-11eb-98c4-2cbb856489aa.png)
![relu_derivative](https://user-images.githubusercontent.com/67945103/120077923-d21c9900-c0e7-11eb-855a-22a94c98a82a.png)
![relu_result](https://user-images.githubusercontent.com/67945103/120077925-d34dc600-c0e7-11eb-92b6-29c6ad6e8faa.png)

We can see **Dying ReLU**.


## GeLU


![gelu](https://user-images.githubusercontent.com/67945103/120077935-db0d6a80-c0e7-11eb-8780-9a6f6eebba61.png)
![gelu_derivative](https://user-images.githubusercontent.com/67945103/120077937-dd6fc480-c0e7-11eb-8700-3157d1ebe25a.png)
![gelu_result](https://user-images.githubusercontent.com/67945103/120077942-dfd21e80-c0e7-11eb-9b68-1418a376d3e0.png)

