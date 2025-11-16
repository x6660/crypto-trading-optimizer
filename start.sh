#!/bin/bash

# AI加密货币交易预测系统 - 快速启动脚本
# 版本: 2.0.0
# 作者: x6660

echo "=================================="
echo "🚀 AI加密货币交易预测系统 v2.0"
echo "=================================="
echo ""

# 检查Python版本
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python版本: $PYTHON_VERSION"
echo ""

# 检查依赖
echo "检查依赖包..."
if [ ! -d "venv" ]; then
    echo "⚠️  未找到虚拟环境，正在创建..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装/更新依赖
echo "检查并安装依赖包..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ 依赖包检查完成"
echo ""

# 检查配置文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到.env配置文件"
    echo "正在从模板创建..."
    cp .env.example .env
    echo "✅ 配置文件已创建: .env"
    echo ""
    echo "⚠️  请编辑 .env 文件，填入你的DeepSeek API密钥："
    echo "   nano .env"
    echo ""
    read -p "按回车键继续（确保已配置API密钥）..."
fi

# 检查端口占用
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口5000已被占用"
    read -p "是否终止占用进程？ (y/n): " kill_process
    if [ "$kill_process" = "y" ]; then
        lsof -ti:5000 | xargs kill -9
        echo "✅ 已终止占用进程"
    else
        echo "❌ 请手动释放端口5000或修改配置"
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "🎯 系统启动中..."
echo "=================================="
echo ""

# 选择启动模式
echo "请选择启动模式："
echo "1) 标准模式 - 使用原有server.py（深度学习模型）"
echo "2) AI增强模式 - 集成DeepSeek AI决策（推荐）"
echo "3) 仅测试环境 - 检查配置和依赖"
echo ""
read -p "请输入选项 (1/2/3): " mode

case $mode in
    1)
        echo ""
        echo "启动标准模式..."
        python3 server.py
        ;;
    2)
        echo ""
        echo "启动AI增强模式..."

        # 检查ai_enhanced_server.py是否存在
        if [ ! -f "ai_enhanced_server.py" ]; then
            echo "⚠️  ai_enhanced_server.py未找到"
            echo "使用标准server.py并集成AI模块..."
            python3 server.py
        else
            python3 ai_enhanced_server.py
        fi
        ;;
    3)
        echo ""
        echo "========== 环境测试 =========="
        echo "Python版本: $PYTHON_VERSION"
        echo ""
        echo "测试导入关键模块..."
        python3 -c "
import sys
print('检查必要的Python包...')
try:
    import flask; print('✅ Flask:', flask.__version__)
except: print('❌ Flask 未安装')

try:
    import pandas; print('✅ Pandas:', pandas.__version__)
except: print('❌ Pandas 未安装')

try:
    import numpy; print('✅ NumPy:', numpy.__version__)
except: print('❌ NumPy 未安装')

try:
    import talib; print('✅ TA-Lib: 已安装')
except: print('❌ TA-Lib 未安装 (需要单独安装C库)')

try:
    import torch; print('✅ PyTorch:', torch.__version__)
except: print('❌ PyTorch 未安装')

try:
    import openai; print('✅ OpenAI SDK:', openai.__version__)
except: print('❌ OpenAI SDK 未安装')

print('')
print('检查配置文件...')
import os
if os.path.exists('.env'):
    print('✅ .env 配置文件存在')
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key and api_key != 'your_api_key_here':
        print('✅ DeepSeek API密钥已配置')
    else:
        print('⚠️  DeepSeek API密钥未配置或使用默认值')
else:
    print('❌ .env 配置文件不存在')

print('')
print('检查模块文件...')
files = ['technical_analysis.py', 'deepseek_ai.py', 'server.py', 'ai_enhanced_server.py']
for f in files:
    if os.path.exists(f):
        print(f'✅ {f}')
    else:
        print(f'❌ {f} 不存在')
"
        echo ""
        echo "========== 测试完成 =========="
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "系统已退出"
echo "=================================="
