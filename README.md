# Activation Functions Test

Test 6 activation functions for training a single node. (1 weight, 1 bias)

- Input : 1.
- Initial weight : -1.
- Initial bias : -1.
- Target : 1.

We can see **Dying ReLU**.  
Also, I tested this because I was curious that since GeLU and swish function has negative derivative (in some part), training could be messed.
So I manipulated initial output (before activation function) to have negative derivatives and could see that it isn't trained.
Actually, it is trained to opposite side.