# Imitation Learning: Behavior Cloning + DAgger in highway-env

Learning to drive by imitation instead of hand-coded rules – a from-scratch 
PyTorch implementation of Behavior Cloning and DAgger in a lightweight 
driving simulator.

## Motivation

This project explores and learns core imitation learning techniques (Behavior Cloning, 
DAgger) as a foundation for robot learning, built during a study-abroad 
semester.

## Results

Evaluated on 200 episodes with seeds 5000–5199, which were not used for collecting the training demonstrations.

| policy    | crash rate | avg return | avg steps |
|-----------|-----------:|-----------:|----------:|
| BC policy |       8.0% |      31.25 |      38.1 |
| expert    |      11.0% |      31.15 |      37.5 |

## Roadmap

**Core pipeline**
- [x] PyTorch fundamentals 
- [x] highway-env observation & action space understood (Kinematics, 
      DiscreteMetaAction)
- [x] Environment setup (for now default highway-v0 config)
- [x] Rule-based expert policy
- [x] Demonstration collection pipeline
- [x] Behavior Cloning implementation
- [x] Evaluation pipeline (BC vs. expert)
- [ ] Distributional shift analysis
- [ ] DAgger implementation
- [ ] BC vs. DAgger comparison

**Future extensions**
- [ ] Finer-grained discrete control (`DiscreteAction`) as a stepping stone
- [ ] Continuous action space
- [ ] Image-based (Grid/Grayscale) observation + CNN policy
- [ ] Sensor fusion (Kalman filter-based state estimation)

## Tech Stack

- Python, PyTorch
- highway-env (Gymnasium)

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`