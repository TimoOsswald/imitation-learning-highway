# Imitation Learning: Behavior Cloning + DAgger in highway-env

Learning to drive by imitation instead of hand-coded rules – a from-scratch 
PyTorch implementation of Behavior Cloning and DAgger in a lightweight 
driving simulator.

## Motivation

This project explores and learns core imitation learning techniques (Behavior Cloning, 
DAgger) as a foundation for robot learning, built during a study-abroad 
semester.

## Roadmap

**Core pipeline**
- [x] PyTorch fundamentals 
- [x] highway-env observation & action space understood (Kinematics, 
      DiscreteMetaAction)
- [ ] Environment setup
- [x] Rule-based expert policy
- [ ] Demonstration collection pipeline
- [ ] Behavior Cloning implementation
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