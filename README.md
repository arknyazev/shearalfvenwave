# STELLGAP Tools

Tools for post-processing results of the STELLGAP, AE3D and FAR3D codes.

## About

STELLGAP computes Shear Alfvén continua in stellarators.

**Method article:** [Shear Alfvén continua in stellarators](https://pubs.aip.org/pop/article/10/8/3217/463091/Shear-Alfven-continua-in-stellarators)  
**STELLGAP source:** [ORNL-Fusion/Stellgap](https://github.com/ORNL-Fusion/Stellgap)

## Installation

```bash
git clone https://github.com/arknyazev/shearalfvenwave.git
cd shearalfvenwave
pip install -e .
```

## Large Data Files

Some examples include large data files stored with Git LFS. 

**To clone with all files:**
```bash
git lfs install  # install GIT LFS if haven't already.
git clone https://github.com/arknyazev/shearalfvenwave.git
```

**To clone without large files (faster):**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/arknyazev/shearalfvenwave.git
```

You can download large files later if needed:
```bash
cd shearalfvenwave
git lfs pull
```

## Usage

Explore the examples directory for typical STELLGAP & AE3D workflows and visualization tools.
