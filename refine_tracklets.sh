#!/bin/bash
set -e

PYTHON_EXEC="/home/yewon/miniconda3/envs/deeptracking/bin/python"

echo "Removing double bboxes..."
$PYTHON_EXEC IDATR/rmv_doub_bbox.py

echo "Generating tracklets..."
$PYTHON_EXEC IDATR/gen_tracklets.py

echo "Refining tracklets..."
$PYTHON_EXEC IDATR/refine_tracklets.py

echo "Creating court files..."
$PYTHON_EXEC IDATR/create_court_file.py

echo "All scripts executed successfully."
