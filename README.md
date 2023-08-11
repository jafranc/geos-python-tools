# geos-python-tools
A set of refactor tools used to post-process GEOS simulations

How to
=======

Log parser
----------

Usage :
```bash
   sudo sshfs -o allow_other,default_permissions $USER@hostname:/path/to/logs /path/to/mountpoint
   python3 main.py --log /path/to/mountpoint/log-to-read
```
