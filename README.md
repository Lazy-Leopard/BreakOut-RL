# Applying RL to Breakout

Applying Reinforcement Learning to basic tasks has been quite a hot topic of interest in the last decade, especially for the second part.One of the basic steps is to begin implementing different algorithms related to it,to basic Games. Classic Arcade Game Enviornments have achieved a special attention towards themselves as a test bed  for these kind of algorithms. My aim is to implement the algorithm(s) to make it/them play the game of Breakout.

## Model(s) under  Implementation :

•	Asynchronous Advantage Actor Critic (A3C)

# A3C (a basic intuition and guide for running)

It's hard to get your state of the art algorithm working,this is because getting any algorithm to work requires some good choices for hyperparameters, and I have to do all of these experiments over my lappy.

THE A3C algorithm can be essentially described as using policy gradients with a function approximator, where the function approximator is a deep neural network and the authors use a clever method to try and ensure the agent explores the state space well.Must admit I am in  love with the idea.With the A3C algorithm,use many agents, all exploring the state space simultaneously. The hope is that the different agents will be in different parts of the state space, and thus give uncorrelated updates to the gradients.

For more better understanding you may refer to [PDF](https://drive.google.com/file/d/1dFfDv-alQs6E_wyRmd3F2ZHKCVxhEIxY/view) , even I recieved cocept and help from lot many places giving each of the link will not be possible,hope this can help you understand the Algorithm.

## Library Requirements 

tensorflow-gpu(1.14.0), numpy, threading, openCV, random, time, gym

The tensorflow version is no hard and fast restriction, you may use any version but will need to take care of the dependencies.










