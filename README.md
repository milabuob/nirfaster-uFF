# nirfaster-uFF

Public repository containing the micro (fast and furious) version of NIRFASTer

- Version: 0.9.6 (beta)
- Authors: Jiaming Cao, MILab@UoB
- License: TBD

This is a compact (aka micro) version of the NIRFASTer with Python interface, providing the most essential functionalities in NIRFASTer. This version is created aiming to be easily integratable with other toolkits, but can also be used on its own.

The toolbox can be run on Linux, Mac, and Windows. To use GPU acceleration, you will need to have a Nvidia card with compute capability higer than `sm_52`, i.e. the GTX9xx series.


## How to Install

1. Clone the main repo, which contains two folders: nirfasteruff (THE source code) and demo (a few demo codes)
2. Depending on your system and Python version, download the appropriate zip file(s) from the appropriate Release, and unzip the contents into the *nirfasteruff* folder
3. You should be good to go

Regardless of your setup, you will need to download the CPU library (cpu-*os*-python*). If your system is CUDA-enabled, you will *also* need to download the appropriate GPU library (gpu-*os*-python*), in addition.

**Special notes to Mac users**

Mac may throw a warning that the file is damaged and need to be moved to Trash. You can bypass this by using command

```bash
xattr -c <your_library>.so
```

## Citation
If you are using our toolbox, please cite the following paper:

Dehghani, Hamid, et al. "Near infrared optical tomography using NIRFAST: Algorithm for numerical model and image reconstruction." Communications in numerical methods in engineering 25.6 (2009): 711-732. doi:10.1002/cnm.1162

## The demos

The provided demo codes shows you all the functionalities the micro version. They are commented in detail and should explain the syntaxes reasonably well.

The head model is adapted from the examples in the NeuroDOT toolbox: https://github.com/WUSTL-ORL/NeuroDOT

## Available key functionalities

- I/O of NIRFAST(er) meshes. This is directly compatible with the Matlab version
- Mesh creation from segmented volumetric data
- Conversion from solid mesh to NIRFASTer mesh
- Fluence calculation (CW/FD)

## Some technical details

The reason for the Mac issue: Mac automatically attaches a quarantine attribute to downloaded files, and the marked files will be checked by the Gatekeeper. Somehow (file I/O, possibly), Apple's gatekeeper is not very happy about my code and refuses to run. This checking can be bypassed by manually removing the quarantine attribute. You can view this by `ls -l@`, and you should see the `com.apple.quarantine` thing.

Speed-critically functions are packed in precompiled libraries, nirfasteruff_cpu and nirfasteruff_cuda. The Linux and Mac versions are statically linked so there is only one file for each library, and no extra dependen is required (need testing). Only limited static linking could be used on Windows (e.g. the CUDA libraries), and consequently the necessary dlls are also included.

I'm compiling for both Python 3.11 (the default version shipped by Anaconda, as of July 2024) and 3.10 (the version cedalion uses). No extra python libraries needed, though of course you will need numpy, scipy, and matplotlib. They should already be available if you use Anaconda python. 

CUDA toolkit used: ver. 12.4, supporting from ```sm_52``` (GTX 9xx series) to ```sm_90a``` (next to be released)

#### Currently tested working good on:

- M1 Macbook Air
- A Intel-chip laptop, no CUDA: Windows 10 and Ubuntu 22.04
- A Intel-chip laptop, with CUDA (sm_75): Windows 10
- A modern desktop (Intel-chip), with CUDA (sm_52): Windows 11 and Ubuntu 22.04
- A private server in lab, with CUDA (sm_70): Ubuntu 22.04
- Special thanks to the developers of Cedalion at TU Berlin, for helping test the toolbox on multiple platforms

#### Potential pitfalls:

- Python by default uses shallow copies, and sometimes fields in the mesh can be accidentally changed because of this. I tried to avoid this by explicitly using deep copies at various places, but undetected problems may still be there
- If a sliced array is fed into the C++ functions, they may throw a type error. This is because the sliced arrays may not be contiguous anymore. Using np.ascontiguousarray(*some_array*, dtype=*some_type*) will solve the problem. I tried to have this safeguard line in most of the high-level functions, but might have overlooked some
- When a C++ function takes a matrix argument, make sure it's actually a matrix. This is especially relevant when you try to feed it with an 1xN matrix. np.atleast2d() can help.
- Will not run on older linux distributions with GLIBC<3.25
