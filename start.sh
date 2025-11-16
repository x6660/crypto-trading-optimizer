#!/bin/bash

echo "========================================"
echo "🚀 加密货币 AI 交易系统 V2.0 Ultimate"
echo "========================================"
echo ""

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python 版本: $python_version"

# 检查依赖
echo "📦 检查依赖安装..."
if ! python3 -c "import talib" 2>/dev/null; then
    echo "❌ TA-Lib 未安装，请先安装 TA-Lib"
    echo "   Ubuntu/Debian: sudo apt-get install ta-lib"
    echo "   macOS: brew install ta-lib"
    exit 1
fi

if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚙️  安装 Python 依赖..."
    pip3 install -r requirements_v2.txt
fi

echo "✓ 所有依赖已安装"
echo ""

# 检查 DeepSeek API Key
if grep -q "你的DeepSeek API Key" server_v2.py; then
    echo "⚠️  请先在 server_v2.py 中配置 DeepSeek API Key"
    echo "   获取地址: https://platform.deepseek.com/api_keys"
    echo ""
    read -p "是否已配置 API Key? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🌐 启动服务器..."
echo "   访问地址: http://localhost:5000"
echo "   按 Ctrl+C 停止服务器"
echo ""
echo "========================================"
echo ""

# 启动服务器
python3 server_v2.py
