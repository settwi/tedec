# Temporal decomposition
Code for decomposing spectrograms using timing information.
Could be e.g. (E, t) spectrograms.
Or some other kind of physically motivated connection.

## Installation
Instructions are for Unix systems.

First, clone the repository:
```bash
git clone https://github.com/settwi/tedec.git
```

Then, set up Python using a virtual environment
    and install the package.
We use a virtual environment to enforce the Python version and ensure
    that packages are installed in the appropriate way.

Use the `uv` tool to do so:
```bash
# The following commands assume that you have changed into the "tedec" directory
# Install `uv` if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart the shell
exec $SHELL

# Create and activate a Python venv in your current directory
uv venv --python 3.13.3
source .venv/bin/activate

# Install the package
uv pip install .

# Install `yaff` v0 if you want to run examples
git clone https://github.com/settwi/yaff.git
cd yaff
git checkout v0
uv pip install .[examples]
# Go back to the tedec directory
cd ..

# To run the examples, while in the venv, run:
python -m jupyter notebook

# To exit the virtual environment:
deactivate
```

If you wish to remove `uv` when you're finished,
    you can delete its files from your home directory:
```bash
rm -rf $HOME/.cache/uv
rm $HOME/.cargo/bin/uv
```

## How to use: see the examples in `examples`!
## _Be sure to run the Jupyter server using the venv created in the setup instructions._
