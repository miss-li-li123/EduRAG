# -*- coding: utf-8 -*-
"""pytest 共享配置：把项目根目录加入 sys.path，保证模块可被 import。"""
import os
import sys

# 本文件位于 <root>/tests/conftest.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
