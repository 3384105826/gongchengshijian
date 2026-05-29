#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安装 python-docx 依赖的脚本
用于支持 Word 文档导出功能
"""

import subprocess
import sys

def install_package():
    """安装 python-docx 包"""
    print("正在安装 python-docx...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
        print("\n✓ python-docx 安装成功！")
        print("\n现在可以运行 pyqt.py 启动程序，使用导出 Word 功能。")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ 安装失败，请检查网络连接或手动运行：pip install python-docx")
        return False
    except Exception as e:
        print(f"\n✗ 发生错误：{e}")
        return False

if __name__ == "__main__":
    install_package()
