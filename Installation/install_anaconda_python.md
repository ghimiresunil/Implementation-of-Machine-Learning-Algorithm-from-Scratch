# How To Install the Anaconda Python on Windows and Linux (Ubuntu and it's Derivatives)

Setting up software requirements (environment) is the first and the most important step in getting started with Machine Learning. It takes a lot of effort to get all of those things ready at times. When we finish preparing for the work environment, we will be halfway there. 

In this file, you’ll learn how to use Anaconda to build up a Python machine learning development environment. These instructions are suitable for both Windows and Linux systems.

* Steps are to install Anaconda in Windows & Ubuntu
* Creating & Working On Conda Environment
* Walkthrough on ML Project

## 1. Introduction
Anaconda is an open-source package manager, environment manager, and distribution of the Python and R programming languages. It is used for data science, machine learning, large-scale data processing, scientific computing, and predictive analytics.

Anaconda aims to simplify package management and deployment. The distribution includes 250 open-source data packages, with over 7,500 more available via the Anaconda repositories suitable for Windows,Linux and MacOS.  It also includes the conda command-line tool and a desktop GUI  called Anaconda Navigator.

### 1.1. Install Anaconda on Windows

* Go to anaconda official website
* [Download](https://www.anaconda.com/products/distribution) based on your Operating set up (64x or 32x) for windows 
* After Anaconda has finished downloading, double-click the _.exe_ file to start the installation process.
* Then, until the installation of Windows is complete, follow the on-screen instructions.
* Don’t forget to add the path to the environmental variable. The benefit is that you can use Anaconda in your Command Prompt, Git Bash, cmder, and so on.
* If you like, you can install Microsoft [VSCode](https://code.visualstudio.com/), but it’s not required.
* Click on Finish
* Open a Command Prompt. If the conda is successfully installed then run conda -V or conda –version in the command prompt and it will pop out the installed version.

### 1.2. Install Anaconda on Ubuntu and it's derivatives

```Anaconda3-2022.05-Linux-x86_64.sh``` is the most recent stable version at the time of writing this post. Check the [Downloads page](https://www.anaconda.com/distribution/) to see whether there is a new version of Anaconda for Python 3 available for download before downloading the installation script.

Downloading the newest Anaconda installer bash script, verifying it, and then running it is the best approach to install Anaconda. To install Anaconda on Ubuntu, follow the steps below:

#### Step 01

Install the following packages if you’re installing Anaconda on a desktop system and want to use the GUI application. 

```
$ sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

#### Step 02

Download the Anaconda installation script with wget

```
$ wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
```

**_Note_**: If you want to work with different version of anaconda, adjust the [version of anaconda](https://repo.anaconda.com/archive/). 

#### Step 03

For verifying the data integrity of the installer with cryptographic hash verification you can use the ```SHA-256``` checksum. You’ll use the sha256sum command along with the filename of the script:

```
$ sha256sum /tmp/Anaconda3-2022.05-Linux-x86_64.sh`
```

You’ll receive output that looks similar to this:

```
Output:
afcd2340039557f7b5e8a8a86affa9ddhsdg887jdfifji988686bds
```

#### Step 04

To begin the installation procedure, run the script

```
$ bash /tmp/Anaconda3-2022.05-Linux-x86_64.sh
```

You’ll receive a welcome mesage as output.  Press `ENTER` to continue and then press `ENTER` to read through the license. Once you complete reading the license, you will be asked for approving the license terms.

```
Output
Do you approve the license terms? [yes|no]
```

Type  `yes`.

Next, you’ll be asked  to choose the location of the installation. You can press `ENTER` to accept the default location, or specify a different location to modify it.
 
Once you are done you will get a thank you message.

#### Step 05

Enter the following bash command to activate the Anaconda installation.

```
$ source ~/.bashrc
```

Once you have done that, you’ll be placed into the default `base` programming environment of Anaconda, and your command prompt will will show base environment. 

``` 
$(base) linus@ubuntu
```

### 1.3. Updating Anaconda 

To update the anaconda to the latest version, open and enter the following command:

```
(base) linus@ubuntu:~$ conda update --all -y
```

### 1.4. Creating & Working on Conda Environment

Anaconda virtual environments let us specify specific package versions. You can specify which version of Python to use for each Anaconda environment you create.

_**Note**_: Name the environment ```venv_first_env``` or some nice and relevant name as per your project.

```
(base) linus@ubuntu:~$ conda create --name venv_first_env python=3.9
```

You’ll receive output with information about what is downloaded and which packages will be installed, and then be prompted to proceed with `y` or `n`. As long as you agree, type `y`.

The `conda` utility will now fetch the packages for the environment and let you know when it’s complete.

You can activate your new environment by typing the following:

```
(base) linus@ubuntu:~$ conda activate venv_first_env
```

With your environment activated, your command prompt prefix will reflect that you are no longer in the `base` environment, but in the new one that you just created.

When you’re ready to deactivate your Anaconda environment, you can do so by typing:

```
(venv_first_env) linus@ubuntu:~$ conda deactivate 
```

With this command, you can see the list of all of the environments you’ve created:

```
(base) linus@ubuntu:~$ conda info --envs
```

When you create environment using  `conda create` , it will come with several default packages. Few examples of then are:


-   `openssl`
-   `pip`
-   `python`
-   `readline`
-   `setuptools`
-   `sqlite`
-   `tk`
 
You might need to add additional package in your environment .

You can add  packages such as `matplotlib` for example, with the following command:

```
(venv_first_env) linus@ubuntu:~$ conda install matplotlib
```


For installing the specific version, you can specify specific version with the following command:
```
(venv_first_env) linus@ubuntu:~$ conda install matplotlib=1.4.3
```


### 1.5. Getting Started With Jupyter Notebook

Jupyter Notebooks are capable of performing data visualization in the same environment and are strong, versatile, and shared. Data scientists may use Jupyter Notebooks to generate and distribute documents ranging from code to full-fledged reports.

You can directly launch Juypter through the terminal using the following command:

#### Command 01:
```
(venv_first_env) linus@ubuntu:~$ jupyter notebook
```

#### command 02:
```
(venv_first_env) linus@ubuntu:~$ jupyter notebook --no-browser --port=8885
```

### 06. Working with Jupyter Noteboook

* create a new notebook, click on the New button on the top right hand corner of the web page and select `Python 3 notebook`. The Python statements are entered in each cell. To execute the Python statements within each cell, press both the `SHIFT` and `ENTER` keys simultaneously. The result will be displayed right below the cell.
* By default, the new notebook will be stored in a file named `Untitled.ipynb`. You can rename the file by clicking on File and Rename menu option at the top
* You can save the notebook by clicking on the `'File'` and `'Save and Checkpoint'` menu options. The notebook will be stored in a file with a `'.ipynb'` extension. You can open the notebook and re-run the program and the results you have saved any time. This powerful feature allows you to share your program and results as well as to reproduce the results generated by others. You can also save the notebook into an HTML format by clicking 'File' followed by 'Download as' options. Note that the IPython notebook is not stored as standard ASCII text files; instead, it is stored in Javascript Object Notation (JSON) file format.

<h1 style="text-align:center"> 🙂 Now we are all done with being ready for Machine learning 🙂 </h1>

