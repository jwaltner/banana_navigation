[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# License

Note that this project is a derivative work and a solution to Udacity's [p1_navigation](https://github.com/udacity/Value-based-methods) project from the Udacity Reinforcement Learning Nanodegree program.  This derivative work complies with the restrictions noted in [LICENSE-udacity](./LICENSE-udacity.md) from Udacity's github project [LICENSE](https://github.com/udacity/Value-based-methods/blob/main/LICENSE.md) for the [Value Based Methods](https://github.com/udacity/Value-based-methods) repository as it is a project operating under the educational use modification and is specifically related to providing a response to Udacity's p1_navigation project which is required to complete the Udacity Reinforcement Learning Nanodegree.  Udacity's license prevents comercial use.  See their license for more details and restrictions on any other derivitave work useage.

Additionally note that the Udacity license appears to be a modified version of the [Creative Commons Attribution-NonCommercial- NoDerivs 3.0](http://creativecommons.org/licenses/by-nc-nd/4.0) license.  Note that the text Udacity's license refers to version 3.0 of the license, but their license links the reader to the 4.0 version through a hyperlink.  This discrepancy between the text and the link from Udacity is retained in this note above.

# Project 1: Bannana Navigation

## Project Details

This project demonstrates how to train an agent which can navigate a walled world of bananas and attempt to collect as many yellow bananas as possible while avoiding the blue bananas.

The environment used for training is a Unity environment which provides the agent with +1 reward for every yellow banana collected and penalizes the agent with a -1 reward for every blue banana collected.  Each episode ends after a fixed amount of time and the environment is considered solved when the agent attains a score of +13 over 100 consecutive episodes.

There are 37 states in this envrionment which includes the agent's velocity as well as features detailing what objects lie on rays extending from the agent's location in the environment which provide the agent with information regarding how far away objects are in the environment.  The agent may use this information to determine the optimal action for any gien state.  The four actions that the agent can choose from are:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Example results from a trained agent operating in the environment:

![Trained Agent][image1]


## Getting Started

1. Use Anaconda to create a Conda environment with the appropriate dependencies installed and make a ipykernel based on this conda environment available for use by jupyter.  Starting from the Udacity base environment [install instructions](https://github.com/udacity/Value-based-methods#dependencies) and then installing the dependencies may result in an environment which can not find all of the versions of the requirements.txt file for the Udacity's `Value-based-methods` project.  If it doesn't work out of the box, then make the suggested modifications to the `requirements.txt` file as noted below.

    ```bash
    # navigate / create a working directory for the project, then execute the following commands
    conda create --name drlnd python=3.6
    conda activate drlnd
    pip install pyvirtualdisplay jupyter jupyterlab matplotlib gymnasium[box2d]

    # Fetch this project and install dependencies
    # Note that this project is the p1-navigation project inside udacity's Value-based-methods repository.
    # git clone https://github.com/udacity/Value-based-methods.git
    git clone https://github.com/jwaltner/banana_navigation
    cd banana_navigation/python

    # Note that the requirements.txt from Udacity's base project called out library versions which were
    # no longer avaialble via pypi.  Therefore, the requirements.txt file here removes all specific versions
    # so that pip install will work.
    pip install .
    
    # Note that installing pytorch with pip from the base python 3.6 install resulted in conflicts.
    # Installing using conda worked, so we install conda separately from the pip install of the requirements
    # text file above.
    conda install pytorch

    # make this conda environment available as a kernel for jupyter.
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in the `p1_navigation/` folder, and unzip (or decompress) the file. Note the location of this application as it will be needed to run the training script in `Navigation.ipynb`

## Instructions

Follow the instructions in `Navigation.ipynb` to train the agent.  Note that with the model provided here which trains from the 37 states provided in the pre-packaged Unity environment, a GPU was not required.

Note that a report on this project is provided in `Report.md`.
