# DJIA Quantum Predictor — 完整部署指南

## 项目结构

```
djia_predictor/
├── app.py                  # 主程序
├── requirements.txt        # 依赖包
├── .streamlit/
│   └── config.toml         # 主题配置
└── README.md               # 本文件
```

---

## 方法一：本地运行（最快）

### 1. 环境准备

```bash
# 确保 Python 3.9+ 已安装
python --version

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

### 4. 输入 API Key

在左侧边栏输入你的 **Anthropic API Key**（从 https://console.anthropic.com 获取）

---

## 方法二：Streamlit Cloud 部署（免费，推荐）

### 步骤 1：上传到 GitHub

```bash
# 初始化 git 仓库
cd djia_predictor
git init
git add .
git commit -m "Initial commit: DJIA Quantum Predictor"

# 创建 GitHub 仓库后推送
git remote add origin https://github.com/你的用户名/djia-predictor.git
git push -u origin main
```

### 步骤 2：在 Streamlit Cloud 部署

1. 访问 https://share.streamlit.io
2. 点击 **"New app"**
3. 选择你的 GitHub 仓库
4. Main file path 填写：`app.py`
5. 点击 **Deploy**

### 步骤 3：配置 API Key（Secrets）

在 Streamlit Cloud 控制台：
- 进入 App Settings → **Secrets**
- 添加以下内容：

```toml
ANTHROPIC_API_KEY = "sk-ant-你的密钥"
```

然后修改 `app.py` 中获取 API Key 的方式：

```python
# 在 render_sidebar() 中替换:
api_key = st.text_input("API Key", type="password")

# 改为:
import os
api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or \
          st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
```

---

## 方法三：Railway 部署（支持自定义域名）

```bash
# 安装 Railway CLI
npm install -g @railway/cli

# 登录
railway login

# 初始化项目
railway init

# 设置环境变量
railway variables set ANTHROPIC_API_KEY=sk-ant-你的密钥

# 部署
railway up
```

在 `railway.toml` 添加：
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"
```

---

## 方法四：Docker 部署

### 创建 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
ENV ANTHROPIC_API_KEY=""

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t djia-predictor .

# 运行容器
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-你的密钥 \
  djia-predictor
```

访问 `http://localhost:8501`

### 使用 docker-compose

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    restart: unless-stopped
```

```bash
ANTHROPIC_API_KEY=sk-ant-你的密钥 docker-compose up -d
```

---

## 方法五：Hugging Face Spaces 部署（免费）

1. 在 https://huggingface.co/spaces 创建新 Space
2. 选择 **Streamlit** 作为 SDK
3. 上传文件
4. 在 Settings → Repository Secrets 添加：
   - `ANTHROPIC_API_KEY`

---

## 常见问题

### Q: API Key 在哪里获取？
访问 https://console.anthropic.com → API Keys → Create Key

### Q: 运行报错 ModuleNotFoundError？
```bash
pip install --upgrade -r requirements.txt
```

### Q: 端口被占用？
```bash
streamlit run app.py --server.port 8502
```

### Q: 如何更新宏观数据？
在 `app.py` 的 `render_sidebar()` 函数中，修改各 slider 的默认值（`value` 参数）以反映最新市场数据。

### Q: 如何添加新的历史事件？
在 `HISTORICAL_EVENTS` 列表中添加新条目：
```python
{"date": "YYYY-MM-DD", "label": "事件名称", "type": "FINANCIAL_CRISIS",
 "impact": -3.0, "decay_months": 12, "region": "US", "desc": "描述"},
```

---

## 模型架构说明

### 因子体系（40+因子）

| 类别 | 因子数 | 说明 |
|------|--------|------|
| 技术指标 | 11 | RSI, MACD, 布林带, ATR, 动量, ROC等 |
| 宏观经济 | 8 | 利率, 通胀, GDP, 失业, 信用利差等 |
| 大宗商品 | 6 | 原油, 天然气, 黄金, 白银, 铜等 |
| 市场情绪 | 5 | VIX, 恐惧贪婪, Put/Call, AAII等 |
| 估值 | 2 | 席勒CAPE, 巴菲特指标 |
| 历史事件 | 9类 | 金融危机/战争/疫情/政策/政治/科技/自然/货币/能源 |

### 历史事件衰减模型

```
当前影响 = 原始冲击 × e^(-λt)
其中:
  λ = ln(2) / (衰减周期/2)  → 半衰期 = 衰减周期的一半
  t = 已过月数
```

### 预测引擎

使用 Claude claude-sonnet-4-20250514 作为非线性集成推理引擎，综合：
1. 动量信号（RSI, MACD, ROC）
2. 均值回归信号（布林带, CAPE）
3. 宏观体制（收益率曲线, 信用利差, 美联储政策）
4. 风险偏好（VIX, 黄金, 美元, Put/Call比率）
5. 商品周期（铜作为经济领先指标, 油价）
6. 情绪反向（极度恐慌=买入信号, 极度贪婪=卖出信号）
7. 历史事件衰减因子（9大类）

---

## 免责声明

本工具仅供学习研究用途，不构成任何投资建议。  
股市预测存在固有不确定性，请勿基于本工具做出实际投资决策。
