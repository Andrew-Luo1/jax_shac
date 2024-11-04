## jax-shac
- An implementation of Short Horizon Actor Critic (Xu; 2022) writen in Jax
- Simulation using the Mujoco MJX simulator

## Results
#### Inverted Pendulum
![inverted_pend-ezgif com-video-to-gif-converter](https://github.com/Andrew-Luo1/jax_shac/assets/22626914/ab14512e-a0cc-4877-80a8-1ec296ebf289)
![image](https://github.com/Andrew-Luo1/jax_shac/assets/22626914/e00d1696-506b-4e91-a26e-aebb0a98403c)

*Run Time*: 1 min jit, 2 min training

*Known Issues*: For some random seeds, you get drift in the cart position.

#### 1 DOF Hopper
![framed_hopper-ezgif com-video-to-gif-converter](https://github.com/Andrew-Luo1/jax_shac/assets/22626914/6edf23b4-6d68-4225-a73a-e473fc0d999a)

![image](https://github.com/Andrew-Luo1/jax_shac/assets/22626914/1e40905d-cc52-465e-90f5-038b46986ff7)

*Run Time*: 1 min jit, 2 min training

*Known Issues*: As seen in the rewards figure, training can be unstable.

## Warning: MJX + Exploding Gradients
- Having great difficulty applying SHAC to get Anymal to walk with default 32-bit precision. (See [Mujoco](https://github.com/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb) for an example with 64-bit precision)
- Hypothesis: it's because quadruped gait is very contact-rich, leading to uninformative gradients.

<img src="https://github.com/Andrew-Luo1/jax_shac/assets/22626914/d774bea2-ef44-4370-8b77-b84594e780a4" width="800">
<img src="https://github.com/Andrew-Luo1/jax_shac/assets/22626914/6262b083-a2dc-4402-ac2e-0d25d76f5cb4" width="800">
<img src="https://github.com/Andrew-Luo1/jax_shac/assets/22626914/a939d83f-2075-4866-8a7a-0893ef892fdf" width="600">

![anymal_vid-ezgif com-video-to-gif-converter](https://github.com/Andrew-Luo1/jax_shac/assets/22626914/47b6561c-1a14-43fb-bfe0-a7d74361e6ec)

*32-step rollout. Ground flashes red when step jacobian is greater than 10e2.*
  
## Setup
- pip install -r requirements.txt
- Add the parent folder of this repository to your PYTHONPATH environment variable.
