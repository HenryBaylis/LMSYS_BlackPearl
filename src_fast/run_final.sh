#!/bin/bash
set -e

# Merge LoRA adapters
python merge_lora.py
python run_gptq.py