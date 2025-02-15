# Language and Input Processing Experiments

## Vision: Breaking Down the Human-AI Communication Barrier

Language and sensory processing are fundamental to human experience. Over millennia, we've evolved sophisticated abilities to read, write, speak, and understand - skills that take years for each human to develop and master. With the advent of neural networks, we're now teaching computers to mirror these deeply human capabilities.

This repository explores different approaches to language and input processing, examining how computers can learn to understand various forms of human communication:
- Handwritten digit recognition (MNIST)
- Natural handwriting recognition (planned)
- Speech recognition (planned)

### Why This Matters

The current bottleneck in human-AI interaction isn't just computational power - it's the speed and richness of our communication. When we interact with AI systems, we're often limited to typing or basic voice commands, losing the nuance and bandwidth that humans naturally use when communicating with each other.

By studying and implementing different input processing techniques, from basic handwriting recognition to more advanced speech processing, we can:
- Understand how neural networks learn to recognize patterns that humans process instinctively
- Explore the parallels between human learning and machine learning
- Work towards reducing the friction in human-AI interaction
- Lay groundwork for more natural and intuitive interfaces

### Project Structure

#### MNIST Digit Recognition
Our journey begins with MNIST - a foundational project in computer vision. While simple by today's standards, it provides crucial insights into:
- Basic pattern recognition principles
- The evolution of neural network architectures (from LeNet-5 to modern approaches)
- The historical context of machine learning (from check processing in the 1980s to modern applications)

[Learn more about our MNIST experiments](docs/mnist.md)

#### Future Explorations
- Advanced handwriting recognition
- Speech processing and recognition
- Visual input processing
- Brain-computer interface simulations

## Getting Started

```bash
# Clone the repository
git clone git@github.com:yourusername/language-input.git

# Create virtual environment
cd language-input
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
