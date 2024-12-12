# nirfaster-uFF

Public repository containing the micro (fast and furious) version of NIRFASTer

- Version: 1.0.0
- Authors: Jiaming Cao, MILab@UoB
- License: BSD

This is a compact (aka micro) version of the NIRFASTer with Python interface, providing the most essential functionalities (that is, forward modeling on standard mesh) in NIRFASTer. This version is created aiming to be easily integratable with other toolkits, but can also be used on its own.

The toolbox can be run on Linux, Mac, and Windows. To use GPU acceleration, you will need to have a Nvidia card with compute capability higer than `sm_52`, i.e. the GTX9xx series.

## How to Install

1. Clone the main repo, which contains two folders: nirfasteruff (THE source code) and demo (a few demo codes)
2. Depending on your system and Python version, download the appropriate zip file(s) from the appropriate Release, and unzip the contents into the *nirfasteruff* folder
3. You should be good to go

Regardless of your setup, you will need to download the CPU library (cpu-*os*-python*), which also includes the mesher binary. If your system is CUDA-enabled, you will *also* need to download the appropriate GPU library (gpu-*os*-python*), in addition.

**Special notes to Mac users**

Mac may throw a warning that the file is damaged and need to be moved to Trash. You can bypass this by using command

```bash
xattr -c <your_library>.so
```

## The full version

The full version of the package can be found at: https://github.com/milabuob/nirfaster-FF

## Citation

A paper on this package is currently in preperation. For now, if you are using our toolbox, please cite the original NIRFAST paper:

Dehghani, Hamid, et al. "Near infrared optical tomography using NIRFAST: Algorithm for numerical model and image reconstruction." Communications in numerical methods in engineering 25.6 (2009): 711-732. doi:10.1002/cnm.1162

## The demos

The provided demo codes shows you all the functionalities the micro version. They are commented in detail and should explain the syntaxes reasonably well.

The head model is adapted from the examples in the NeuroDOT toolbox: https://github.com/WUSTL-ORL/NeuroDOT

## Available key functionalities

- I/O of NIRFAST(er) meshes. This is directly compatible with the Matlab version
- Mesh creation from segmented volumetric data
- Conversion from solid mesh to NIRFASTer mesh
- Fluence calculation (CW/FD)

## Changelog

1.0.0
- Added support for Python 3.12
- CGAL mesher separated from the CPU library as a standalone application
- Fixed a bug in gen_intmat, which leads to incorrect data.tomesh() result when xgrid and ygrid have different resolutions
- Fixed a bug which causes crashes when mesh has only one source
- Number of OMP threads in the CPU solvers can now be set using function nirfasteruff.utils.get_nthread()
- GPU solver performance improved
- More efficient source vector calculation
- docstring reformatted for better readability
- Stricter error handling (consistent with the full version), which raises errors instead of printing a message

## Some technical details

The reason for the Mac issue: Mac automatically attaches a quarantine attribute to downloaded files, and the marked files will be checked by the Gatekeeper. Somehow (file I/O, possibly), Apple's gatekeeper is not very happy about my code and refuses to run. This checking can be bypassed by manually removing the quarantine attribute. You can view this by `ls -l@`, and you should see the `com.apple.quarantine` thing.

Speed-critically functions are packed in precompiled libraries, nirfasteruff_cpu and nirfasteruff_cuda. The Linux and Mac versions are statically linked so there is only one file for each library, and no extra dependen is required (need testing). Only limited static linking could be used on Windows (e.g. the CUDA libraries), and consequently the necessary dlls are also included.


#### Potential pitfalls:

- Python by default uses shallow copies, and sometimes fields in the mesh can be accidentally changed because of this. I tried to avoid this by explicitly using deep copies at various places, but undetected problems may still be there
- If a sliced array is fed into the C++ functions, they may throw a type error. This is because the sliced arrays may not be contiguous anymore. Using np.ascontiguousarray(*some_array*, dtype=*some_type*) will solve the problem. I tried to have this safeguard line in most of the high-level functions, but might have overlooked some
- When a C++ function takes a matrix argument, make sure it's actually a matrix. This is especially relevant when you try to feed it with an 1xN matrix. np.atleast2d() can help.
- Will not run on older linux distributions with GLIBC<3.25
