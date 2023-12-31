# drq_jax
Jax Implementation of Data-regularized Q (DrQ)

(It's my lecture project of Reinforcement Learning :)

### How to run?

`python drq.py cfg=walker_walk train_seed=0`

### Compare with [official implementation](https://github.com/denisyarats/drq)
#### Performance
![new_performance_curve_](https://github.com/mantle2048/drq_jax/assets/37854077/0f4a1c40-5b54-49cb-bffc-ec07de655c2d)

#### Wall clock time 
![new_time_curve](https://github.com/mantle2048/drq_jax/assets/37854077/f355aa2e-75a0-453e-a9a8-c084663ed86b)

### Disclaimers
Running the code requires ≈38 GB GPU memory.

As I can access large memory GPUs, so I did not implement a memory-efficient replay buffer for image observations.

Leave it for future work (下次一定!)
