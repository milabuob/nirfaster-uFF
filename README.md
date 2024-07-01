# nirfaster-uFF

Public repository containing the micro (fast and furious) version of NIRFASTer

- Version: 0.9.5 (beta)
- Authors: Jiaming Cao, MILab@UoB
- License: TBD

This is a compact (aka micro) version of the NIRFASTer with Python interface, providing the most essential functionalities in NIRFASTer. This version is created aiming to be easily integratable with other toolkits, but can also be used on its own.

The toolbox can be run on Linux, Mac, and Windows. To use GPU acceleration, you will need to have a Nvidia card with compute capability higer than `sm_52`, i.e. the GTX9xx series.

## How to Install

1. Clone the main repo, which contains THE python file and some demo codes
2. Depending on your system and Python version, download the appropriate zip file(s) from the Release, and unzip the contents into the main nirfaster-uFF folder

Regardless of your setup, you will need to download the CPU library (cpu-*os*-python*). If your system is CUDA-enabled, you will *also* need to download the appropriate GPU library (gpu-*os*-python*), in addition.

## The demos

The provided demo codes shows you all the functionalities the micro version. They are commented in detail and should explain the syntaxes reasonably well.

The head model is adapted from the examples in the NeuroDOT toolbox: https://github.com/WUSTL-ORL/NeuroDOT

## Some technical details

Speed-critically functions are packed in precompiled libraries, nirfasteruff_cpu and nirfasteruff_cuda. The Linux and Mac versions are statically linked so there is only one file for each library, and no extra dependen is required (need testing). Only limited static linking could be used on Windows (e.g. the CUDA libraries), and consequently the necessary dlls are also included.

I'm compiling for both Python 3.11 (the default version shipped by Anaconda, as of July 2024) and 3.10 (the version cedalion uses). No extra python libraries needed, though of course you will need numpy, scipy, and matplotlib. They should already be available if you use Anaconda python. 

CUDA toolkit used: ver. 12.4, supporting from sm_52 (GTX 9xx series) to sm_90a (next to be released)

#### Currently tested working good on:

- M1 Macbook Air
- A Intel-chip laptop, no CUDA: Windows 10 and Ubuntu 22.04
- A Intel-chip laptop, with CUDA (sm_75): Windows 10
- A modern desktop (Intel-chip), with CUDA (sm_52): Windows 11 and Ubuntu 22.04
- A private server in lab, with CUDA (sm_70): Ubuntu 22.04

#### Further testing needed

- Non-stock C++ libraries used on Mac. Statically linked, but may have overlooked some libs. Need to check if it runs on other devices. Cross-compiled for Intel Macs as well, but not tested
- Should run OK on up-to-date Linux machines

#### Potential pitfalls:

- Python by default uses shallow copies, and sometimes fields in the mesh can be accidentally changed because of this. I tried to avoid this by explicitly using deep copies at various places, but undetected problems may still be there
- If a sliced array is fed into the C++ functions, they may throw a type error. This is because the sliced arrays may not be contiguous anymore. Using np.ascontiguousarray(*some_array*, dtype=*some_type*) will solve the problem. I tried to have this safeguard line in most of the high-level functions, but might have overlooked some
- When a C++ function takes a matrix argument, make sure it's actually a matrix. This is especially relevant when you try to feed it with an 1xN matrix. np.atleast2d() can help.
- Will not run on older linux distributions with GLIBC<3.25
