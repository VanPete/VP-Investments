# Migration Guide: Version 1.0 to Version 2.0

## Overview

This document provides information about migrating from VP Investments Version 1.0 (this repository state) to Version 2.0.

## Version 1.0 Archive

This repository currently contains **Version 1.0** of VP Investments, which is now archived. All files have been marked with version information to clearly identify this as the v1.0 codebase.

### What's Been Preserved

- All v1.0 source code
- Complete project structure
- Configuration files
- Documentation
- Web dashboard (v1.0)

### Version Markers

The following files contain version identification:
- `VP Investments.py` - Main orchestrator (v1.0)
- `run_all.py` - Runner script (v1.0)
- `config/config.py` - Configuration (v1.0)
- `VERSION.txt` - Version information file
- `README.md` - Updated with archive notice

## Version 2.0

Version 2.0 is under active development in VSCode and will be pushed to this repository when ready.

### Transition Plan

1. **Current State**: This repository contains v1.0 (archived)
2. **Next Step**: Repository owner will push v2.0 codebase
3. **v1.0 Access**: This version will be available via git history/tags

### For Repository Owner

To push Version 2.0:

```bash
# Option 1: Direct replacement (if you want v2.0 on main branch)
# Backup v1.0 first by creating a tag
git tag v1.0.0-archive
git push origin v1.0.0-archive

# Then push v2.0 to main
git checkout main
# (copy your v2.0 files)
git add .
git commit -m "Release Version 2.0"
git push origin main

# Option 2: Branch strategy
# Create v1-archive branch to preserve v1.0
git checkout -b v1-archive
git push origin v1-archive

# Then work on main for v2.0
git checkout main
# (copy your v2.0 files)
git add .
git commit -m "Release Version 2.0"
git push origin main
```

### Preserving v1.0

To preserve v1.0 access, consider:
1. Creating a `v1.0.0-archive` git tag
2. Creating a `v1-archive` branch
3. Updating the main branch with v2.0

This ensures users can still access v1.0 if needed while v2.0 becomes the primary version.

## Questions?

For questions about migration or accessing specific versions, contact the repository owner.
