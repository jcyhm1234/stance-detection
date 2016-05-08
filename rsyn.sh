#!/bin/bash  
# use this script to send code to remote comp, you can run there
# uses lot of resources on laptop, making laptop unusable
# rsync only sends modified files over
rsync -r -v src/ rahulrh@dots.cs.utexas.edu:projects/stance-detection/src/