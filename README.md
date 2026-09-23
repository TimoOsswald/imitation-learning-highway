# Imitation Learning: Behavior Cloning + DAgger in highway-env

Learning to drive by imitation instead of hand-coded rules – a from-scratch 
PyTorch implementation of Behavior Cloning and DAgger in a lightweight 
driving simulator.

## Motivation

This project explores and learns core imitation learning techniques (Behavior Cloning, 
DAgger) as a foundation for robot learning, built during a study-abroad 
semester.

## Status

Work in progress. Currently working through PyTorch fundamentals 
(tensors, datasets/dataloaders, neural network basics) before implementing 
the actual training pipeline.

## Roadmap

- [x] PyTorch basics (tensors, autograd fundamentals)
- [ ] highway-env setup & exploration
- [ ] Demonstration collection pipeline
- [ ] Behavior Cloning implementation
- [ ] Distributional shift analysis
- [ ] DAgger implementation
- [ ] BC vs. DAgger comparison

## Tech Stack

- Python, PyTorch
- highway-env (Gymnasium)

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`