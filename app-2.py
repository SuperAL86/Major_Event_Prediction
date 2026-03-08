"""
DJIA Quantum Predictor - Streamlit Edition
多因子预测模型：技术指标 + 宏观经济 + 大宗商品 + 历史事件因子
Author: Multi-Factor Research Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import anthropic
import json
from datetime import datetime, timedelta
import math

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DJIA Quantum Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700;800&display=swap');

    .stApp { background-color: #040810; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #040810 0%, #0a1628 50%, #040810 100%);
        border: 1px solid #0f2040;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 28px; font-weight: 800; letter-spacing: 4px;
        color: #e2e8f0; margin: 0;
    }
    .hero-subtitle {
        font-size: 11px; color: #334155; letter-spacing: 2px; margin-top: 6px;
    }

    /* Metric cards */
    .metric-card {
        background: #080e1c;
        border: 1px solid #0f2040;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value { font-size: 22px; font-weight: 800; letter-spacing: 1px; }
    .metric-label { font-size: 10px; color: #475569; letter-spacing: 1px; margin-top: 3px; }

    /* Prediction card */
    .pred-card {
        background: linear-gradient(135deg, #080e1c, #0a1f10);
        border: 1px solid #10b98133;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
    }
    .pred-price { font-size: 52px; font-weight: 800; color: #10b981; letter-spacing: 2px; }
    .pred-bearish { font-size: 52px; font-weight: 800; color: #ef4444; letter-spacing: 2px; }
    .pred-neutral { font-size: 52px; font-weight: 800; color: #f59e0b; letter-spacing: 2px; }

    /* Factor pill */
    .bull-factor {
        background: #052011; border-left: 3px solid #10b981;
        padding: 6px 10px; border-radius: 4px;
        font-size: 11px; color: #6ee7b7; margin: 3px 0;
    }
    .bear-factor {
        background: #1a0505; border-left: 3px solid #ef4444;
        padding: 6px 10px; border-radius: 4px;
        font-size: 11px; color: #fca5a5; margin: 3px 0;
    }
    .risk-box {
        background: #0e0a00; border: 1px solid #78350f;
        border-radius: 6px; padding: 10px 14px; margin-top: 10px;
    }

    /* Event type badge */
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 10px; font-weight: 700; letter-spacing: 1px;
    }

    /* Sidebar */
    .stSidebar { background-color: #040c18 !important; }
    section[data-testid="stSidebar"] { background-color: #040c18; border-right: 1px solid #0f2040; }
    section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8) !important;
        color: white !important;
        border: 1px solid #3b82f6 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        box-shadow: 0 0 20px rgba(59,130,246,0.3) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover { box-shadow: 0 0 30px rgba(59,130,246,0.5) !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #040810; border-bottom: 1px solid #0f2040; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        color: #334155 !important; font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important; letter-spacing: 1px !important;
        background: transparent !important; border: none !important;
        padding: 10px 18px !important;
    }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom: 2px solid #60a5fa !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] { color: #3b82f6; }

    /* Section headers */
    .section-header {
        font-size: 11px; font-weight: 700; color: #60a5fa;
        letter-spacing: 3px; margin: 0 0 12px 0;
        text-transform: uppercase;
    }

    /* Disclaimer */
    .disclaimer {
        background: #040810; border-top: 1px solid #0a1628;
        padding: 12px 20px; font-size: 9px; color: #1e3a5f;
        border-radius: 0 0 8px 8px; margin-top: 20px;
    }

    /* Dataframe */
    .stDataFrame { border: 1px solid #0f2040; border-radius: 6px; }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #080e1c !important;
        color: #94a3b8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        border: 1px solid #0f2040 !important;
    }

    hr { border-color: #0f2040 !important; }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1: HISTORICAL EVENTS DATABASE
# 100年重大历史事件数据库（不限国家/类型）
# Impact: -5 (catastrophic) to +5 (transformative positive)
# Decay: how many months until impact fades (市场冲击衰减月数)
# ─────────────────────────────────────────────
HISTORICAL_EVENTS = [
    # ══════════════════════════════════════════
    # 金融危机 / FINANCIAL CRISES
    # ══════════════════════════════════════════
    {"date": "1929-10-24", "label": "黑色星期四", "type": "FINANCIAL_CRISIS",
     "impact": -5.0, "decay_months": 36, "region": "US",
     "desc": "纽约股市崩盘，道指单日跌11%，大萧条序幕"},
    {"date": "1929-10-29", "label": "黑色星期二", "type": "FINANCIAL_CRISIS",
     "impact": -5.0, "decay_months": 40, "region": "US",
     "desc": "股市彻底崩溃，道指跌89%，大萧条正式开始"},
    {"date": "1931-09-21", "label": "英国放弃金本位", "type": "FINANCIAL_CRISIS",
     "impact": -3.0, "decay_months": 12, "region": "UK",
     "desc": "英镑危机，国际货币体系动荡"},
    {"date": "1933-03-06", "label": "美国银行业假日", "type": "FINANCIAL_CRISIS",
     "impact": -2.5, "decay_months": 3, "region": "US",
     "desc": "罗斯福宣布银行假日，恐慌暂停"},
    {"date": "1987-10-19", "label": "黑色星期一", "type": "FINANCIAL_CRISIS",
     "impact": -5.0, "decay_months": 6, "region": "GLOBAL",
     "desc": "道指单日暴跌22.6%，史上最大单日跌幅"},
    {"date": "1989-12-29", "label": "日本泡沫顶峰", "type": "FINANCIAL_CRISIS",
     "impact": -2.0, "decay_months": 24, "region": "JAPAN",
     "desc": "日经指数见顶38957，随后十年熊市"},
    {"date": "1992-09-16", "label": "黑色星期三英镑危机", "type": "FINANCIAL_CRISIS",
     "impact": -2.5, "decay_months": 6, "region": "UK",
     "desc": "索罗斯做空英镑，英国被迫退出ERM"},
    {"date": "1994-12-20", "label": "墨西哥比索危机", "type": "FINANCIAL_CRISIS",
     "impact": -2.0, "decay_months": 8, "region": "LATAM",
     "desc": "比索贬值50%，新兴市场传染效应"},
    {"date": "1997-07-02", "label": "亚洲金融危机", "type": "FINANCIAL_CRISIS",
     "impact": -4.0, "decay_months": 18, "region": "ASIA",
     "desc": "泰铢崩溃，席卷东南亚，韩国、印尼告急"},
    {"date": "1998-08-17", "label": "俄罗斯债务违约", "type": "FINANCIAL_CRISIS",
     "impact": -3.5, "decay_months": 8, "region": "RUSSIA",
     "desc": "主权债务违约，LTCM对冲基金濒临崩溃"},
    {"date": "2000-03-10", "label": "互联网泡沫顶峰", "type": "FINANCIAL_CRISIS",
     "impact": -4.5, "decay_months": 30, "region": "US",
     "desc": "纳斯达克5048见顶，科技股暴跌80%"},
    {"date": "2007-08-09", "label": "次贷危机爆发", "type": "FINANCIAL_CRISIS",
     "impact": -3.5, "decay_months": 18, "region": "US",
     "desc": "BNP巴黎银行冻结基金，信用市场冻结"},
    {"date": "2008-03-17", "label": "贝尔斯登崩溃", "type": "FINANCIAL_CRISIS",
     "impact": -3.0, "decay_months": 12, "region": "US",
     "desc": "美国第五大投行被摩根大通救援收购"},
    {"date": "2008-09-15", "label": "雷曼兄弟破产", "type": "FINANCIAL_CRISIS",
     "impact": -5.0, "decay_months": 18, "region": "GLOBAL",
     "desc": "史上最大企业破产，全球金融体系濒临崩溃"},
    {"date": "2010-05-06", "label": "闪崩Flash Crash", "type": "FINANCIAL_CRISIS",
     "impact": -3.0, "decay_months": 2, "region": "US",
     "desc": "道指午后骤跌1000点，算法交易引发"},
    {"date": "2010-05-02", "label": "希腊债务危机", "type": "FINANCIAL_CRISIS",
     "impact": -3.0, "decay_months": 24, "region": "EUROPE",
     "desc": "欧债危机爆发，欧元区解体恐慌"},
    {"date": "2015-08-24", "label": "中国股灾", "type": "FINANCIAL_CRISIS",
     "impact": -3.5, "decay_months": 8, "region": "CHINA",
     "desc": "A股熔断，道指单日跌千点，全球共振"},
    {"date": "2020-03-16", "label": "新冠市场崩溃", "type": "FINANCIAL_CRISIS",
     "impact": -5.0, "decay_months": 4, "region": "GLOBAL",
     "desc": "道指单日跌2997点，史上最大点数跌幅"},
    {"date": "2022-01-05", "label": "加息恐慌熊市", "type": "FINANCIAL_CRISIS",
     "impact": -3.0, "decay_months": 12, "region": "US",
     "desc": "美联储激进加息引发科技股熊市"},
    {"date": "2023-03-10", "label": "硅谷银行倒闭", "type": "FINANCIAL_CRISIS",
     "impact": -2.5, "decay_months": 4, "region": "US",
     "desc": "美国史上第二大银行倒闭，银行业危机蔓延"},

    # ══════════════════════════════════════════
    # 战争与地缘冲突 / WAR & GEOPOLITICAL
    # ══════════════════════════════════════════
    {"date": "1939-09-01", "label": "二战爆发", "type": "WAR",
     "impact": -3.0, "decay_months": 60, "region": "GLOBAL",
     "desc": "德国入侵波兰，欧洲股市暴跌，美市初期下跌"},
    {"date": "1940-05-10", "label": "德军入侵西欧", "type": "WAR",
     "impact": -3.5, "decay_months": 12, "region": "EUROPE",
     "desc": "法国沦陷，英国危机，美股急跌"},
    {"date": "1941-12-07", "label": "珍珠港事件", "type": "WAR",
     "impact": -4.0, "decay_months": 6, "region": "US",
     "desc": "日本偷袭，美国参战，道指周内跌6%"},
    {"date": "1945-05-08", "label": "欧战胜利VE Day", "type": "WAR",
     "impact": 3.0, "decay_months": 6, "region": "GLOBAL",
     "desc": "纳粹投降，战争结束，市场大涨"},
    {"date": "1945-08-15", "label": "日本投降", "type": "WAR",
     "impact": 3.5, "decay_months": 8, "region": "GLOBAL",
     "desc": "二战结束，战后繁荣预期"},
    {"date": "1950-06-25", "label": "朝鲜战争爆发", "type": "WAR",
     "impact": -2.5, "decay_months": 8, "region": "ASIA",
     "desc": "冷战升温，市场短暂下跌"},
    {"date": "1956-10-23", "label": "苏伊士危机/匈牙利", "type": "WAR",
     "impact": -2.0, "decay_months": 4, "region": "GLOBAL",
     "desc": "双重地缘政治危机，油价飙升"},
    {"date": "1962-10-16", "label": "古巴导弹危机", "type": "WAR",
     "impact": -4.0, "decay_months": 3, "region": "GLOBAL",
     "desc": "核战边缘，道指13天跌10%"},
    {"date": "1967-06-05", "label": "六日战争", "type": "WAR",
     "impact": -2.0, "decay_months": 4, "region": "MIDEAST",
     "desc": "中东战争，石油供应威胁"},
    {"date": "1973-10-06", "label": "赎罪日战争+石油危机", "type": "WAR",
     "impact": -5.0, "decay_months": 18, "region": "MIDEAST",
     "desc": "OPEC禁运，油价暴涨400%，道指跌45%"},
    {"date": "1979-12-27", "label": "苏联入侵阿富汗", "type": "WAR",
     "impact": -2.0, "decay_months": 12, "region": "GLOBAL",
     "desc": "冷战再度升温，粮食禁运，通胀加剧"},
    {"date": "1982-04-02", "label": "马岛战争", "type": "WAR",
     "impact": -1.0, "decay_months": 3, "region": "UK",
     "desc": "英阿冲突，区域性影响"},
    {"date": "1990-08-02", "label": "海湾战争爆发", "type": "WAR",
     "impact": -3.5, "decay_months": 6, "region": "MIDEAST",
     "desc": "伊拉克入侵科威特，油价飙升，道指跌20%"},
    {"date": "1991-01-17", "label": "海湾战争开打", "type": "WAR",
     "impact": 2.5, "decay_months": 3, "region": "MIDEAST",
     "desc": "军事行动成功，不确定性消除，市场急涨"},
    {"date": "1999-03-24", "label": "科索沃战争", "type": "WAR",
     "impact": -1.5, "decay_months": 4, "region": "EUROPE",
     "desc": "北约空袭，欧洲局势紧张"},
    {"date": "2001-09-11", "label": "9/11恐怖袭击", "type": "WAR",
     "impact": -5.0, "decay_months": 8, "region": "US",
     "desc": "道指当周重开后跌14.3%，VIX飙升"},
    {"date": "2003-03-20", "label": "伊拉克战争", "type": "WAR",
     "impact": -2.0, "decay_months": 6, "region": "MIDEAST",
     "desc": "开战市场下跌，随后因不确定性消除反弹"},
    {"date": "2006-07-12", "label": "黎以战争", "type": "WAR",
     "impact": -1.0, "decay_months": 2, "region": "MIDEAST",
     "desc": "中东局势再度紧张"},
    {"date": "2014-03-18", "label": "俄罗斯吞并克里米亚", "type": "WAR",
     "impact": -2.0, "decay_months": 8, "region": "RUSSIA",
     "desc": "制裁开始，乌克兰危机，能源市场波动"},
    {"date": "2022-02-24", "label": "俄乌战争全面爆发", "type": "WAR",
     "impact": -4.0, "decay_months": 18, "region": "EUROPE",
     "desc": "欧洲最大军事冲突，能源危机，通胀激化"},
    {"date": "2023-10-07", "label": "以哈战争", "type": "WAR",
     "impact": -2.0, "decay_months": 12, "region": "MIDEAST",
     "desc": "哈马斯袭击以色列，中东局势重燃"},
    {"date": "2025-04-15", "label": "中东局势升级", "type": "WAR",
     "impact": -1.5, "decay_months": 6, "region": "MIDEAST",
     "desc": "伊朗制裁升级，红海航运中断延续"},

    # ══════════════════════════════════════════
    # 疫情与健康危机 / PANDEMIC & HEALTH
    # ══════════════════════════════════════════
    {"date": "1918-09-01", "label": "西班牙流感", "type": "PANDEMIC",
     "impact": -3.0, "decay_months": 18, "region": "GLOBAL",
     "desc": "全球5000万死亡，战时管控限制市场反应"},
    {"date": "1957-09-01", "label": "亚洲流感", "type": "PANDEMIC",
     "impact": -1.5, "decay_months": 6, "region": "GLOBAL",
     "desc": "H2N2流感，全球100万死亡"},
    {"date": "1968-07-01", "label": "香港流感", "type": "PANDEMIC",
     "impact": -1.0, "decay_months": 6, "region": "GLOBAL",
     "desc": "H3N2，全球100万死亡，市场轻微波动"},
    {"date": "2003-03-01", "label": "SARS疫情", "type": "PANDEMIC",
     "impact": -2.5, "decay_months": 6, "region": "ASIA",
     "desc": "亚洲经济受冲击，旅游消费股重挫"},
    {"date": "2009-04-01", "label": "H1N1猪流感", "type": "PANDEMIC",
     "impact": -1.5, "decay_months": 4, "region": "GLOBAL",
     "desc": "全球大流行，但金融危机后市场已处于低位"},
    {"date": "2014-10-01", "label": "埃博拉危机", "type": "PANDEMIC",
     "impact": -1.5, "decay_months": 4, "region": "GLOBAL",
     "desc": "西非爆发，恐慌传至美国，消费股受压"},
    {"date": "2020-01-20", "label": "新冠首例美国确诊", "type": "PANDEMIC",
     "impact": -1.0, "decay_months": 2, "region": "US",
     "desc": "初期市场温和反应"},
    {"date": "2020-02-24", "label": "新冠全球扩散恐慌", "type": "PANDEMIC",
     "impact": -4.5, "decay_months": 4, "region": "GLOBAL",
     "desc": "意大利疫情爆发，市场大规模抛售启动"},
    {"date": "2020-03-11", "label": "WHO宣布大流行", "type": "PANDEMIC",
     "impact": -5.0, "decay_months": 3, "region": "GLOBAL",
     "desc": "历史最快熊市，道指月跌37%"},
    {"date": "2020-11-09", "label": "辉瑞疫苗90%有效率", "type": "PANDEMIC",
     "impact": 4.5, "decay_months": 6, "region": "GLOBAL",
     "desc": "疫苗突破，道指单日涨1500点"},
    {"date": "2021-11-26", "label": "Omicron变种", "type": "PANDEMIC",
     "impact": -2.5, "decay_months": 3, "region": "GLOBAL",
     "desc": "新变种恐慌，感恩节后市场大跌"},

    # ══════════════════════════════════════════
    # 经济政策 / ECONOMIC POLICY
    # ══════════════════════════════════════════
    {"date": "1933-03-09", "label": "罗斯福新政开启", "type": "ECON_POLICY",
     "impact": 4.0, "decay_months": 24, "region": "US",
     "desc": "银行法案+社会保障+公共就业，大萧条转机"},
    {"date": "1944-07-22", "label": "布雷顿森林协议", "type": "ECON_POLICY",
     "impact": 3.0, "decay_months": 12, "region": "GLOBAL",
     "desc": "战后国际货币体系奠基，美元霸权确立"},
    {"date": "1971-08-15", "label": "尼克松关闭黄金窗口", "type": "ECON_POLICY",
     "impact": -2.5, "decay_months": 12, "region": "GLOBAL",
     "desc": "布雷顿森林体系终结，美元与黄金脱钩"},
    {"date": "1979-10-06", "label": "沃尔克冲击加息", "type": "ECON_POLICY",
     "impact": -3.0, "decay_months": 18, "region": "US",
     "desc": "美联储暴力加息至20%，打击通胀也引发衰退"},
    {"date": "1981-08-13", "label": "里根减税法案", "type": "ECON_POLICY",
     "impact": 4.0, "decay_months": 24, "region": "US",
     "desc": "史上最大规模减税，供给侧经济学，牛市起点"},
    {"date": "1982-08-17", "label": "美联储降息转向", "type": "ECON_POLICY",
     "impact": 4.5, "decay_months": 12, "region": "US",
     "desc": "沃尔克降息，通胀受控，1982年大牛市"},
    {"date": "1985-09-22", "label": "广场协议", "type": "ECON_POLICY",
     "impact": 2.0, "decay_months": 12, "region": "GLOBAL",
     "desc": "G5压低美元，日元欧元升值，全球资产重估"},
    {"date": "1993-01-01", "label": "欧洲单一市场", "type": "ECON_POLICY",
     "impact": 2.5, "decay_months": 12, "region": "EUROPE",
     "desc": "欧盟单一市场启动，贸易壁垒消除"},
    {"date": "1994-01-01", "label": "NAFTA生效", "type": "ECON_POLICY",
     "impact": 2.0, "decay_months": 12, "region": "US",
     "desc": "北美自由贸易区，供应链整合"},
    {"date": "1999-01-01", "label": "欧元启动", "type": "ECON_POLICY",
     "impact": 2.0, "decay_months": 8, "region": "EUROPE",
     "desc": "欧元区建立，全球货币格局重塑"},
    {"date": "2001-12-11", "label": "中国加入WTO", "type": "ECON_POLICY",
     "impact": 3.0, "decay_months": 24, "region": "CHINA",
     "desc": "全球化加速，制造业重心转移，通胀长期压低"},
    {"date": "2008-10-08", "label": "全球央行协调降息", "type": "ECON_POLICY",
     "impact": 2.5, "decay_months": 4, "region": "GLOBAL",
     "desc": "七国央行联合降息50bp，危机中的协调"},
    {"date": "2009-02-17", "label": "美国7870亿刺激法案", "type": "ECON_POLICY",
     "impact": 3.5, "decay_months": 12, "region": "US",
     "desc": "奥巴马经济刺激，市场企稳转机"},
    {"date": "2010-11-03", "label": "QE2量化宽松", "type": "ECON_POLICY",
     "impact": 3.0, "decay_months": 8, "region": "US",
     "desc": "美联储购债6000亿，流动性泛滥"},
    {"date": "2012-07-26", "label": "德拉吉不惜一切", "type": "ECON_POLICY",
     "impact": 4.0, "decay_months": 12, "region": "EUROPE",
     "desc": "欧央行承诺无限购债，欧债危机化解"},
    {"date": "2013-05-22", "label": "缩减恐慌Taper Tantrum", "type": "ECON_POLICY",
     "impact": -2.5, "decay_months": 6, "region": "GLOBAL",
     "desc": "美联储暗示缩减QE，新兴市场大跌"},
    {"date": "2017-12-22", "label": "特朗普减税法案", "type": "ECON_POLICY",
     "impact": 3.0, "decay_months": 12, "region": "US",
     "desc": "企业税从35%降至21%，企业盈利飙升"},
    {"date": "2018-03-01", "label": "中美贸易战开始", "type": "ECON_POLICY",
     "impact": -3.0, "decay_months": 18, "region": "GLOBAL",
     "desc": "钢铝关税开始，全球贸易体系动荡"},
    {"date": "2020-03-27", "label": "CARES法案2.2万亿", "type": "ECON_POLICY",
     "impact": 4.0, "decay_months": 6, "region": "US",
     "desc": "史上最大经济刺激，直升机撒钱"},
    {"date": "2021-11-03", "label": "美联储Taper开始", "type": "ECON_POLICY",
     "impact": -1.5, "decay_months": 6, "region": "US",
     "desc": "缩减购债，货币政策转向信号"},
    {"date": "2022-03-16", "label": "美联储激进加息周期", "type": "ECON_POLICY",
     "impact": -4.0, "decay_months": 18, "region": "GLOBAL",
     "desc": "连续加息至5.25-5.5%，全球资产重估"},
    {"date": "2024-09-18", "label": "美联储开始降息", "type": "ECON_POLICY",
     "impact": 2.5, "decay_months": 8, "region": "US",
     "desc": "降息50bp，流动性改善预期"},
    {"date": "2025-04-02", "label": "特朗普对等关税", "type": "ECON_POLICY",
     "impact": -4.0, "decay_months": 12, "region": "GLOBAL",
     "desc": "全面关税，道指暴跌，贸易战2.0"},

    # ══════════════════════════════════════════
    # 政治事件 / POLITICAL EVENTS
    # ══════════════════════════════════════════
    {"date": "1932-11-08", "label": "罗斯福当选总统", "type": "POLITICAL",
     "impact": 2.5, "decay_months": 6, "region": "US",
     "desc": "新政承诺，市场底部转机"},
    {"date": "1945-04-12", "label": "罗斯福逝世", "type": "POLITICAL",
     "impact": -1.5, "decay_months": 2, "region": "US",
     "desc": "战时总统骤逝，短暂不确定性"},
    {"date": "1948-11-02", "label": "杜鲁门意外连任", "type": "POLITICAL",
     "impact": 1.0, "decay_months": 3, "region": "US",
     "desc": "选举意外，市场震荡后稳定"},
    {"date": "1949-10-01", "label": "中华人民共和国成立", "type": "POLITICAL",
     "impact": -1.0, "decay_months": 12, "region": "CHINA",
     "desc": "冷战格局深化，亚洲局势剧变"},
    {"date": "1955-09-26", "label": "艾森豪威尔心脏病", "type": "POLITICAL",
     "impact": -2.5, "decay_months": 2, "region": "US",
     "desc": "总统健康危机，道指单日跌6.5%"},
    {"date": "1960-11-08", "label": "肯尼迪当选", "type": "POLITICAL",
     "impact": 1.0, "decay_months": 4, "region": "US",
     "desc": "新边疆政策，市场审慎乐观"},
    {"date": "1963-11-22", "label": "肯尼迪遇刺", "type": "POLITICAL",
     "impact": -3.0, "decay_months": 3, "region": "US",
     "desc": "道指单日跌3%，次日强势反弹"},
    {"date": "1968-04-04", "label": "马丁路德金遇刺", "type": "POLITICAL",
     "impact": -2.0, "decay_months": 3, "region": "US",
     "desc": "社会动荡，城市骚乱，市场恐慌"},
    {"date": "1972-02-21", "label": "尼克松访华", "type": "POLITICAL",
     "impact": 2.0, "decay_months": 6, "region": "GLOBAL",
     "desc": "中美破冰，国际局势缓和"},
    {"date": "1974-08-09", "label": "尼克松辞职", "type": "POLITICAL",
     "impact": -1.5, "decay_months": 4, "region": "US",
     "desc": "水门事件终结，政治不确定性消除"},
    {"date": "1980-11-04", "label": "里根当选", "type": "POLITICAL",
     "impact": 3.0, "decay_months": 12, "region": "US",
     "desc": "里根经济学，减税降监管，牛市催化"},
    {"date": "1989-06-04", "label": "天安门事件", "type": "POLITICAL",
     "impact": -2.0, "decay_months": 8, "region": "CHINA",
     "desc": "中国政治危机，国际制裁威胁"},
    {"date": "1989-11-09", "label": "柏林墙倒塌", "type": "POLITICAL",
     "impact": 3.5, "decay_months": 12, "region": "GLOBAL",
     "desc": "冷战终结信号，全球化加速"},
    {"date": "1991-12-26", "label": "苏联解体", "type": "POLITICAL",
     "impact": 2.5, "decay_months": 12, "region": "GLOBAL",
     "desc": "冷战结束，和平红利，全球化大时代"},
    {"date": "1997-07-01", "label": "香港回归", "type": "POLITICAL",
     "impact": 0.5, "decay_months": 4, "region": "CHINA",
     "desc": "历史性交接，市场平稳过渡"},
    {"date": "2000-11-07", "label": "小布什vs戈尔大选悬念", "type": "POLITICAL",
     "impact": -1.5, "decay_months": 3, "region": "US",
     "desc": "历史最接近选举，持续36天不确定"},
    {"date": "2008-11-04", "label": "奥巴马当选", "type": "POLITICAL",
     "impact": 1.5, "decay_months": 4, "region": "US",
     "desc": "历史性时刻，变革预期，金融危机背景"},
    {"date": "2016-06-23", "label": "英国脱欧公投", "type": "POLITICAL",
     "impact": -3.5, "decay_months": 18, "region": "EUROPE",
     "desc": "英镑暴跌8%，欧洲股市崩盘"},
    {"date": "2016-11-08", "label": "特朗普意外当选", "type": "POLITICAL",
     "impact": 3.0, "decay_months": 12, "region": "US",
     "desc": "减税+基建预期，道指直线拉升"},
    {"date": "2020-11-03", "label": "拜登当选", "type": "POLITICAL",
     "impact": 2.0, "decay_months": 6, "region": "US",
     "desc": "政策不确定性消除，市场上涨"},
    {"date": "2021-01-06", "label": "国会山骚乱", "type": "POLITICAL",
     "impact": -1.0, "decay_months": 2, "region": "US",
     "desc": "民主冲击，市场短暂恐慌"},
    {"date": "2024-11-05", "label": "特朗普再次当选", "type": "POLITICAL",
     "impact": 3.0, "decay_months": 8, "region": "US",
     "desc": "减税+去监管预期，道指和加密货币暴涨"},
    {"date": "2025-01-20", "label": "特朗普就任第二任期", "type": "POLITICAL",
     "impact": 1.5, "decay_months": 6, "region": "US",
     "desc": "政策执行期开始，市场初期乐观"},

    # ══════════════════════════════════════════
    # 科技革命 / TECH BREAKTHROUGHS
    # ══════════════════════════════════════════
    {"date": "1957-10-04", "label": "苏联发射Sputnik", "type": "TECH",
     "impact": -1.5, "decay_months": 6, "region": "GLOBAL",
     "desc": "太空竞赛，冷战科技军备"},
    {"date": "1969-07-20", "label": "人类首次登月", "type": "TECH",
     "impact": 1.5, "decay_months": 4, "region": "US",
     "desc": "科技里程碑，美国实力彰显"},
    {"date": "1971-11-15", "label": "英特尔推出微处理器", "type": "TECH",
     "impact": 1.0, "decay_months": 12, "region": "US",
     "desc": "个人电脑时代序幕"},
    {"date": "1981-08-12", "label": "IBM PC发布", "type": "TECH",
     "impact": 1.5, "decay_months": 12, "region": "US",
     "desc": "个人电脑革命开始"},
    {"date": "1991-08-06", "label": "万维网公开发布", "type": "TECH",
     "impact": 2.0, "decay_months": 24, "region": "GLOBAL",
     "desc": "互联网时代序幕，改变一切"},
    {"date": "1995-08-09", "label": "网景IPO", "type": "TECH",
     "impact": 2.0, "decay_months": 12, "region": "US",
     "desc": "互联网热潮正式启动"},
    {"date": "1997-05-11", "label": "深蓝击败卡斯帕罗夫", "type": "TECH",
     "impact": 1.0, "decay_months": 4, "region": "US",
     "desc": "AI里程碑，科技股兴奋"},
    {"date": "2007-01-09", "label": "iPhone发布", "type": "TECH",
     "impact": 2.5, "decay_months": 18, "region": "GLOBAL",
     "desc": "移动互联网革命，苹果股价长期受益"},
    {"date": "2009-01-03", "label": "比特币诞生", "type": "TECH",
     "impact": 0.5, "decay_months": 6, "region": "GLOBAL",
     "desc": "区块链革命种子，初期市场忽视"},
    {"date": "2016-03-15", "label": "AlphaGo击败李世石", "type": "TECH",
     "impact": 1.5, "decay_months": 8, "region": "GLOBAL",
     "desc": "深度学习AI突破，科技股受益"},
    {"date": "2022-11-30", "label": "ChatGPT发布", "type": "TECH",
     "impact": 3.5, "decay_months": 24, "region": "GLOBAL",
     "desc": "生成式AI革命，科技股长期牛市催化"},
    {"date": "2023-03-14", "label": "GPT-4发布", "type": "TECH",
     "impact": 2.5, "decay_months": 12, "region": "GLOBAL",
     "desc": "AI能力跨越式提升，算力需求爆炸"},
    {"date": "2024-01-10", "label": "比特币ETF获批", "type": "TECH",
     "impact": 2.0, "decay_months": 8, "region": "US",
     "desc": "机构资金流入加密货币"},
    {"date": "2025-01-27", "label": "DeepSeek冲击", "type": "TECH",
     "impact": -2.0, "decay_months": 4, "region": "GLOBAL",
     "desc": "中国AI模型低成本崛起，英伟达单日跌17%"},

    # ══════════════════════════════════════════
    # 自然灾害 / NATURAL DISASTERS
    # ══════════════════════════════════════════
    {"date": "1906-04-18", "label": "旧金山大地震", "type": "NATURAL",
     "impact": -2.0, "decay_months": 6, "region": "US",
     "desc": "重建成本，保险股重创"},
    {"date": "1986-04-26", "label": "切尔诺贝利核事故", "type": "NATURAL",
     "impact": -2.0, "decay_months": 8, "region": "GLOBAL",
     "desc": "核能股暴跌，苏联经济雪上加霜"},
    {"date": "1995-01-17", "label": "阪神大地震", "type": "NATURAL",
     "impact": -2.0, "decay_months": 6, "region": "JAPAN",
     "desc": "日本经济重创，保险和建筑业震荡"},
    {"date": "2004-12-26", "label": "印度洋海啸", "type": "NATURAL",
     "impact": -1.0, "decay_months": 3, "region": "ASIA",
     "desc": "人道主义灾难，旅游和保险股受压"},
    {"date": "2005-08-29", "label": "卡特里娜飓风", "type": "NATURAL",
     "impact": -2.0, "decay_months": 6, "region": "US",
     "desc": "能源设施破坏，油价上涨，保险损失"},
    {"date": "2011-03-11", "label": "日本311大地震海啸", "type": "NATURAL",
     "impact": -3.5, "decay_months": 8, "region": "JAPAN",
     "desc": "福岛核危机，全球供应链断裂"},
    {"date": "2017-08-25", "label": "哈维飓风", "type": "NATURAL",
     "impact": -1.5, "decay_months": 3, "region": "US",
     "desc": "能源基础设施受损，石油精炼受阻"},
    {"date": "2023-02-06", "label": "土耳其叙利亚地震", "type": "NATURAL",
     "impact": -1.0, "decay_months": 4, "region": "MIDEAST",
     "desc": "区域性影响，重建需求"},

    # ══════════════════════════════════════════
    # 货币/汇率冲击 / CURRENCY CRISES
    # ══════════════════════════════════════════
    {"date": "1923-11-15", "label": "德国恶性通胀", "type": "CURRENCY",
     "impact": -3.0, "decay_months": 12, "region": "EUROPE",
     "desc": "德国马克崩溃，汇率1美元兑4.2万亿马克"},
    {"date": "1967-11-18", "label": "英镑大幅贬值", "type": "CURRENCY",
     "impact": -2.0, "decay_months": 6, "region": "UK",
     "desc": "英镑贬值14%，国际货币秩序动荡"},
    {"date": "1973-03-01", "label": "固定汇率体系终结", "type": "CURRENCY",
     "impact": -1.5, "decay_months": 8, "region": "GLOBAL",
     "desc": "布雷顿森林彻底崩溃，浮动汇率时代"},
    {"date": "1994-12-20", "label": "比索危机", "type": "CURRENCY",
     "impact": -2.5, "decay_months": 8, "region": "LATAM",
     "desc": "墨西哥比索崩溃，拉美传染"},
    {"date": "1997-07-02", "label": "泰铢崩溃", "type": "CURRENCY",
     "impact": -4.0, "decay_months": 18, "region": "ASIA",
     "desc": "亚洲金融危机触发，货币危机席卷"},
    {"date": "2001-12-20", "label": "阿根廷披索危机", "type": "CURRENCY",
     "impact": -2.0, "decay_months": 12, "region": "LATAM",
     "desc": "主权违约，银行挤兑，社会动荡"},
    {"date": "2015-08-11", "label": "人民币突然贬值", "type": "CURRENCY",
     "impact": -2.5, "decay_months": 6, "region": "CHINA",
     "desc": "央行意外贬值2%，全球新兴市场暴跌"},
    {"date": "2022-09-28", "label": "英国迷你预算英镑危机", "type": "CURRENCY",
     "impact": -2.5, "decay_months": 4, "region": "UK",
     "desc": "英镑跌至历史低位，英债崩溃，央行紧急干预"},
    {"date": "2023-05-01", "label": "美元债务上限危机", "type": "CURRENCY",
     "impact": -1.5, "decay_months": 3, "region": "US",
     "desc": "美国债务违约风险，全球金融体系紧张"},

    # ══════════════════════════════════════════
    # 能源危机 / ENERGY CRISES
    # ══════════════════════════════════════════
    {"date": "1973-10-17", "label": "OPEC石油禁运", "type": "ENERGY",
     "impact": -5.0, "decay_months": 18, "region": "GLOBAL",
     "desc": "油价暴涨400%，全球经济滞胀"},
    {"date": "1979-01-16", "label": "伊朗革命石油危机", "type": "ENERGY",
     "impact": -4.0, "decay_months": 18, "region": "GLOBAL",
     "desc": "第二次石油危机，油价再涨150%"},
    {"date": "1986-01-01", "label": "油价崩溃", "type": "ENERGY",
     "impact": -2.0, "decay_months": 12, "region": "GLOBAL",
     "desc": "沙特增产，油价跌60%，能源股重挫"},
    {"date": "1990-08-02", "label": "海湾战争油价暴涨", "type": "ENERGY",
     "impact": -2.5, "decay_months": 6, "region": "GLOBAL",
     "desc": "科威特被占，油价一夜翻倍"},
    {"date": "2008-07-11", "label": "油价冲140美元", "type": "ENERGY",
     "impact": -2.0, "decay_months": 6, "region": "GLOBAL",
     "desc": "油价历史高位，通胀压力极度"},
    {"date": "2014-06-20", "label": "油价暴跌", "type": "ENERGY",
     "impact": -1.5, "decay_months": 12, "region": "GLOBAL",
     "desc": "页岩油革命+OPEC价格战，油价跌75%"},
    {"date": "2020-04-20", "label": "原油期货负价格", "type": "ENERGY",
     "impact": -2.5, "decay_months": 4, "region": "GLOBAL",
     "desc": "WTI跌至-37美元，史无前例的能源崩溃"},
    {"date": "2022-03-01", "label": "俄乌冲突能源危机", "type": "ENERGY",
     "impact": -3.0, "decay_months": 12, "region": "EUROPE",
     "desc": "欧洲天然气危机，能源通胀失控"},
]

# ─────────────────────────────────────────────
# SECTION 2: EVENT TYPE METADATA
# ─────────────────────────────────────────────
EVENT_TYPES = {
    "FINANCIAL_CRISIS": {"label": "金融危机", "color": "#ef4444", "icon": "💥", "base_weight": -0.85},
    "WAR":              {"label": "战争冲突", "color": "#f97316", "icon": "⚔️", "base_weight": -0.60},
    "PANDEMIC":         {"label": "疫情",     "color": "#8b5cf6", "icon": "🦠", "base_weight": -0.70},
    "ECON_POLICY":      {"label": "经济政策", "color": "#3b82f6", "icon": "📋", "base_weight":  0.50},
    "POLITICAL":        {"label": "政治事件", "color": "#06b6d4", "icon": "🏛️", "base_weight":  0.20},
    "TECH":             {"label": "科技突破", "color": "#10b981", "icon": "💡", "base_weight":  0.55},
    "NATURAL":          {"label": "自然灾害", "color": "#6b7280", "icon": "🌋", "base_weight": -0.45},
    "CURRENCY":         {"label": "货币危机", "color": "#fbbf24", "icon": "💱", "base_weight": -0.65},
    "ENERGY":           {"label": "能源危机", "color": "#f59e0b", "icon": "⚡", "base_weight": -0.55},
}

# ─────────────────────────────────────────────
# SECTION 3: COMPUTE ACTIVE EVENT FACTOR SCORES
# For a given reference date, compute the decayed
# impact score of each event type
# ─────────────────────────────────────────────
def compute_event_factors(reference_date_str: str) -> dict:
    """
    For each event type, compute a composite score based on:
    1. Historical base rate impact (statistical average)
    2. Currently active events (within decay window)
    3. Exponential decay of recent events
    """
    ref_date = pd.to_datetime(reference_date_str)
    scores = {et: 0.0 for et in EVENT_TYPES}
    active_events = []

    for evt in HISTORICAL_EVENTS:
        evt_date = pd.to_datetime(evt["date"])
        months_elapsed = (ref_date - evt_date).days / 30.44
        if 0 <= months_elapsed <= evt["decay_months"]:
            # Exponential decay: impact * e^(-lambda * t)
            # where lambda = ln(2) / (decay_months/2) => half-life = decay_months/2
            half_life = max(evt["decay_months"] / 2, 1)
            lam = math.log(2) / half_life
            decayed_impact = evt["impact"] * math.exp(-lam * months_elapsed)
            scores[evt["type"]] += decayed_impact
            if abs(decayed_impact) > 0.1:
                active_events.append({
                    **evt,
                    "months_elapsed": round(months_elapsed, 1),
                    "decayed_impact": round(decayed_impact, 3),
                })

    # Clip to reasonable range
    for et in scores:
        scores[et] = max(-5.0, min(5.0, scores[et]))

    return scores, active_events


# ─────────────────────────────────────────────
# SECTION 4: DJIA SYNTHETIC DATA
# ─────────────────────────────────────────────
@st.cache_data
def build_price_series():
    anchors = [
        ("2020-01-02", 28868), ("2020-02-28", 25409), ("2020-03-23", 18213),
        ("2020-04-30", 24346), ("2020-06-30", 25813), ("2020-08-31", 28430),
        ("2020-10-30", 26502), ("2020-11-30", 29639), ("2020-12-31", 30606),
        ("2021-03-31", 32981), ("2021-06-30", 34503), ("2021-09-30", 33844),
        ("2021-12-31", 36338), ("2022-03-31", 34678), ("2022-06-30", 30775),
        ("2022-09-30", 28726), ("2022-12-30", 33147), ("2023-03-31", 33274),
        ("2023-06-30", 34408), ("2023-09-29", 33507), ("2023-12-29", 37090),
        ("2024-03-28", 39807), ("2024-06-28", 39118), ("2024-09-30", 42330),
        ("2024-11-29", 44910), ("2024-12-31", 42544), ("2025-03-31", 42280),
        ("2025-06-30", 43535), ("2025-09-30", 42330), ("2025-12-31", 43000),
        ("2026-03-07", 43200),
    ]
    records = []
    rng = np.random.default_rng(42)
    for i in range(len(anchors) - 1):
        d1, v1 = pd.to_datetime(anchors[i][0]), anchors[i][1]
        d2, v2 = pd.to_datetime(anchors[i + 1][0]), anchors[i + 1][1]
        dates = pd.bdate_range(d1, d2)
        n = len(dates)
        for j, dt in enumerate(dates):
            t = j / max(n - 1, 1)
            noise = rng.normal(0, v1 * 0.006)
            price = v1 + (v2 - v1) * t + noise
            records.append({"date": dt, "close": round(price, 2)})

    df = pd.DataFrame(records).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change() * 100
    df["high"] = df["close"] * (1 + np.abs(np.sin(np.arange(len(df)) * 1.3)) * 0.007)
    df["low"] = df["close"] * (1 - np.abs(np.cos(np.arange(len(df)) * 1.7)) * 0.007)

    # ── Technical indicators ──
    for w in [5, 10, 20, 50, 200]:
        df[f"sma{w}"] = df["close"].rolling(w).mean()
    k12, k26 = 2 / 13, 2 / 27
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    roll20 = df["close"].rolling(20)
    df["bb_mid"] = roll20.mean()
    df["bb_std"] = roll20.std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["momentum10"] = df["close"].pct_change(10) * 100
    df["roc5"] = df["close"].pct_change(5) * 100
    df["roc20"] = df["close"].pct_change(20) * 100
    df["vol20"] = df["return"].rolling(20).std() * np.sqrt(252)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["vol_momentum"] = df["return"].rolling(5).std() / df["return"].rolling(20).std()
    df["price_vs_sma20"] = (df["close"] / df["sma20"] - 1) * 100
    df["price_vs_sma50"] = (df["close"] / df["sma50"] - 1) * 100
    df["price_vs_sma200"] = (df["close"] / df["sma200"] - 1) * 100

    return df.dropna(subset=["rsi", "macd"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# SECTION 5: CLAUDE PREDICTION ENGINE
# ─────────────────────────────────────────────
def run_prediction(df: pd.DataFrame, macro: dict, event_scores: dict,
                   active_events: list, scenario: str, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    latest = df.iloc[-1]
    recent_5 = df.tail(5)["return"].tolist()

    # Format event factors for prompt
    event_factor_str = "\n".join([
        f"  {EVENT_TYPES[et]['icon']} {EVENT_TYPES[et]['label']} ({et}): {score:+.3f}"
        for et, score in event_scores.items()
    ])
    active_str = "\n".join([
        f"  [{e['type']}] {e['date']} {e['label']} | 原始冲击:{e['impact']:+.1f} | 衰减后:{e['decayed_impact']:+.3f} | 经过:{e['months_elapsed']}月"
        for e in sorted(active_events, key=lambda x: abs(x['decayed_impact']), reverse=True)[:10]
    ]) if active_events else "  (当前无活跃事件)"

    prompt = f"""You are an elite quantitative analyst running an ensemble multi-factor prediction model for the Dow Jones Industrial Average (DJIA). Use ALL factors below with sophisticated non-linear interactions.

═══════════════════════════════════════
CURRENT DJIA: {latest['close']:.2f} | DATE: {latest['date'].strftime('%Y-%m-%d')}
═══════════════════════════════════════

▸ TECHNICAL FACTORS:
  RSI(14): {latest['rsi']:.1f} ({'OVERBOUGHT>70' if latest['rsi']>70 else 'OVERSOLD<30' if latest['rsi']<30 else 'NEUTRAL'})
  MACD: {latest['macd']:.1f} | Signal: {latest['macd_signal']:.1f} | Histogram: {latest['macd_hist']:.1f}
  Bollinger Band Position: {latest['bb_pos']:.3f} (0=lower, 1=upper)
  BB Width: {latest['bb_width']:.4f} ({'SQUEEZE<0.04' if latest['bb_width']<0.04 else 'WIDE'})
  10d Momentum: {latest['momentum10']:.2f}%
  ROC-5d: {latest['roc5']:.2f}% | ROC-20d: {latest['roc20']:.2f}%
  ATR(14): {latest['atr14']:.0f}pts | Vol20d(ann): {latest['vol20']:.1f}%
  Price vs SMA20: {latest['price_vs_sma20']:.2f}%
  Price vs SMA50: {latest['price_vs_sma50']:.2f}%
  Price vs SMA200: {latest['price_vs_sma200']:.2f}%
  Last 5 daily returns: {[f'{r:.2f}%' for r in recent_5]}

▸ MACRO-ECONOMIC FACTORS:
  Fed Funds Rate: {macro['fed_rate']}% | 10Y Yield: {macro['yield_10y']}% | 2Y Yield: {macro['yield_2y']}%
  Yield Curve (10Y-2Y): {macro['yield_10y']-macro['yield_2y']:+.2f}% ({'INVERTED=RECESSION RISK' if macro['yield_10y']-macro['yield_2y']<0 else 'NORMAL'})
  CPI YoY: {macro['cpi']}% | PCE YoY: {macro['pce']}%
  Unemployment: {macro['unemployment']}% | GDP Growth: {macro['gdp']}%
  Credit Spread (IG): {macro['credit_spread']}% | M2 Growth: {macro['m2']}%
  Dollar Index (DXY): {macro['dxy']} | Margin Debt: ${macro['margin_debt']}B

▸ COMMODITIES & PRECIOUS METALS:
  WTI Crude: ${macro['wti']}/bbl | Brent: ${macro['brent']}/bbl | Nat Gas: ${macro['natgas']}/MMBtu
  Gold: ${macro['gold']}/oz ({'ELEVATED=RISK-OFF' if macro['gold']>2500 else 'NORMAL'}) | Silver: ${macro['silver']}/oz
  Copper: ${macro['copper']}/lb ({'STRONG=GROWTH' if macro['copper']>4 else 'WEAK=SLOWDOWN'})

▸ SENTIMENT & VALUATION:
  VIX: {macro['vix']} ({'EXTREME FEAR>30' if macro['vix']>30 else 'HIGH FEAR>20' if macro['vix']>20 else 'COMPLACENT'})
  Shiller CAPE: {macro['cape']}x ({'DANGEROUSLY HIGH>35' if macro['cape']>35 else 'HIGH>30' if macro['cape']>30 else 'FAIR'})
  Buffett Indicator: {macro['buffett']}% | Put/Call Ratio: {macro['put_call']}
  Fear & Greed Index: {macro['fear_greed']}/100 | AAII Bulls: {macro['aaii_bull']}%

▸ HISTORICAL EVENT FACTORS (decayed impact scores):
{event_factor_str}

▸ CURRENTLY ACTIVE HISTORICAL EVENTS (within decay window):
{active_str}

▸ USER SCENARIO INPUT:
{scenario if scenario.strip() else "(no additional scenario)"}

═══════════════════════════════════════
MODELING INSTRUCTIONS:
1. Apply non-linear factor interactions: high VIX + negative MACD + inverted yield curve = multiplicatively bearish
2. Use contrarian logic where appropriate: extreme fear (VIX>35, F&G<20) = mean reversion bullish signal
3. Weight historical event factors by both magnitude and recency
4. Consider regime context: are we in risk-on, risk-off, transitional?
5. Ensemble approach: combine momentum, mean-reversion, macro, sentiment, event factors
6. Day-to-day moves for DJIA typically range ±0.3% to ±1.5%; large moves (>2%) require multiple confirming bearish signals

Respond ONLY in this exact JSON (no markdown, no preamble):
{{
  "prediction": <float: next trading day DJIA close>,
  "confidence_low": <float: 80% CI lower>,
  "confidence_high": <float: 80% CI upper>,
  "expected_return_pct": <float: predicted % return>,
  "direction": <"BULLISH"|"BEARISH"|"NEUTRAL">,
  "confidence": <int: 0-100>,
  "regime": <"RISK_ON"|"RISK_OFF"|"TRANSITIONAL">,
  "top_bullish": [<str>, <str>, <str>],
  "top_bearish": [<str>, <str>, <str>],
  "key_risks": [<str>, <str>],
  "factor_scores": {{
    "technical": <float -10 to 10>,
    "momentum": <float -10 to 10>,
    "macro": <float -10 to 10>,
    "valuation": <float -10 to 10>,
    "sentiment": <float -10 to 10>,
    "commodities": <float -10 to 10>,
    "event_financial_crisis": <float -10 to 10>,
    "event_war": <float -10 to 10>,
    "event_pandemic": <float -10 to 10>,
    "event_policy": <float -10 to 10>,
    "event_political": <float -10 to 10>,
    "event_tech": <float -10 to 10>,
    "event_energy": <float -10 to 10>,
    "event_currency": <float -10 to 10>
  }},
  "narrative": <str: 4-5 sentences explaining the prediction logic>,
  "week_bias": <"BULLISH"|"BEARISH"|"NEUTRAL">,
  "month_bias": <"BULLISH"|"BEARISH"|"NEUTRAL">,
  "quarter_bias": <"BULLISH"|"BEARISH"|"NEUTRAL">
}}"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()
    # Extract JSON
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])


# ─────────────────────────────────────────────
# SECTION 6: SIDEBAR — INPUTS
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="section-header">⚙ 系统配置</p>', unsafe_allow_html=True)

        api_key = st.text_input(
            "Anthropic API Key", type="password",
            placeholder="sk-ant-...",
            help="从 console.anthropic.com 获取"
        )
        st.divider()

        st.markdown('<p class="section-header">📡 宏观因子输入</p>', unsafe_allow_html=True)

        with st.expander("💵 货币政策", expanded=True):
            fed_rate     = st.slider("联邦基金利率 (%)",   0.0, 8.0, 4.33, 0.25)
            yield_10y    = st.slider("10年期国债收益率 (%)", 1.0, 8.0, 4.21, 0.01)
            yield_2y     = st.slider("2年期国债收益率 (%)",  0.5, 8.0, 3.99, 0.01)
            credit_spread = st.slider("投资级信用利差 (%)",  0.3, 5.0, 1.15, 0.05)
            m2           = st.slider("M2货币增速 (%)",      -5.0, 25.0, 3.8, 0.1)

        with st.expander("📊 实体经济"):
            cpi          = st.slider("CPI同比 (%)",         0.0, 12.0, 2.8, 0.1)
            pce          = st.slider("PCE同比 (%)",         0.0, 10.0, 2.5, 0.1)
            unemployment = st.slider("失业率 (%)",          2.0, 15.0, 4.1, 0.1)
            gdp          = st.slider("GDP增速-年化 (%)",   -15.0, 10.0, 2.3, 0.1)
            margin_debt  = st.slider("融资余额 (十亿$)",   400, 1000, 780, 5)

        with st.expander("🛢 大宗商品 & 贵金属"):
            wti    = st.slider("WTI原油 ($/桶)",   20, 150, 71, 1)
            brent  = st.slider("布伦特原油 ($/桶)", 20, 160, 75, 1)
            natgas = st.slider("天然气 ($/MMBtu)", 1.0, 15.0, 3.85, 0.05)
            gold   = st.slider("黄金 ($/盎司)",  1200, 4000, 2912, 5)
            silver = st.slider("白银 ($/盎司)",    10, 60, 32, 1)
            copper = st.slider("铜 ($/磅)",       1.5, 7.0, 4.52, 0.05)

        with st.expander("😱 情绪 & 估值"):
            vix        = st.slider("VIX恐慌指数",       10, 90, 22, 1)
            cape       = st.slider("席勒CAPE (x)",        8, 50, 35, 1)
            buffett    = st.slider("巴菲特指标 (%)",      50, 220, 188, 1)
            fear_greed = st.slider("恐惧贪婪指数 (0-100)", 0, 100, 38, 1)
            put_call   = st.slider("Put/Call比率",       0.4, 2.0, 0.82, 0.01)
            aaii_bull  = st.slider("AAII看多比例 (%)",    5, 80, 32, 1)
            dxy        = st.slider("美元指数 (DXY)",      80, 120, 104, 1)

        macro = dict(
            fed_rate=fed_rate, yield_10y=yield_10y, yield_2y=yield_2y,
            credit_spread=credit_spread, m2=m2, cpi=cpi, pce=pce,
            unemployment=unemployment, gdp=gdp, margin_debt=margin_debt,
            wti=wti, brent=brent, natgas=natgas,
            gold=gold, silver=silver, copper=copper,
            vix=vix, cape=cape, buffett=buffett,
            fear_greed=fear_greed, put_call=put_call,
            aaii_bull=aaii_bull, dxy=dxy,
        )
        return api_key, macro


# ─────────────────────────────────────────────
# SECTION 7: CHART HELPERS
# ─────────────────────────────────────────────
CHART_BG    = "#040810"
CHART_GRID  = "#0f2040"
CHART_FONT  = dict(family="JetBrains Mono, monospace", color="#94a3b8")

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=CHART_FONT, margin=dict(l=40, r=20, t=30, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)
    return fig


def price_chart(df, pred=None, days=90):
    sub = df.tail(days)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.03)

    # Price + BBands
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["bb_upper"], mode="lines",
        line=dict(color="#334155", width=1, dash="dot"), name="BB上轨", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["bb_lower"], mode="lines",
        line=dict(color="#334155", width=1, dash="dot"), name="BB下轨",
        fill="tonexty", fillcolor="rgba(51,65,85,0.08)", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["sma20"],
        line=dict(color="#f59e0b", width=1.2, dash="dash"), name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["sma50"],
        line=dict(color="#8b5cf6", width=1.2, dash="dash"), name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"],
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)", name="DJIA"), row=1, col=1)

    if pred:
        next_date = sub["date"].iloc[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[sub["date"].iloc[-1], next_date],
            y=[sub["close"].iloc[-1], pred["prediction"]],
            mode="lines+markers", line=dict(color="#10b981", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond"), name="预测"), row=1, col=1)
        fig.add_hrect(y0=pred["confidence_low"], y1=pred["confidence_high"],
            fillcolor="rgba(16,185,129,0.06)", line_width=0, row=1, col=1)

    # MACD
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in sub["macd_hist"]]
    fig.add_trace(go.Bar(x=sub["date"], y=sub["macd_hist"],
        marker_color=colors, name="MACD柱", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["macd"],
        line=dict(color="#60a5fa", width=1.5), name="MACD", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["macd_signal"],
        line=dict(color="#f97316", width=1.2), name="信号线", showlegend=False), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["rsi"],
        line=dict(color="#a78bfa", width=1.5), name="RSI(14)", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ef4444", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#10b981", width=1, dash="dot"), row=3, col=1)

    fig.update_yaxes(title_text="DJIA", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="MACD", row=2, col=1, title_font_size=10)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100], title_font_size=10)
    fig.update_layout(height=600, title="DJIA 技术分析图表")
    return style_fig(fig)


def factor_bar_chart(factor_scores: dict):
    label_map = {
        "technical": "技术指标", "momentum": "价格动量", "macro": "宏观经济",
        "valuation": "估值", "sentiment": "市场情绪", "commodities": "大宗商品",
        "event_financial_crisis": "事件:金融危机", "event_war": "事件:战争",
        "event_pandemic": "事件:疫情", "event_policy": "事件:经济政策",
        "event_political": "事件:政治", "event_tech": "事件:科技",
        "event_energy": "事件:能源", "event_currency": "事件:货币危机",
    }
    labels = [label_map.get(k, k) for k in factor_scores]
    values = list(factor_scores.values())
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, marker_opacity=0.85,
        text=[f"{v:+.1f}" for v in values],
        textposition="outside", textfont=dict(size=10)
    ))
    fig.add_vline(x=0, line=dict(color="#334155", width=1.5))
    fig.update_layout(height=460, title="因子得分分解",
                      xaxis=dict(range=[-12, 12], title="得分"))
    return style_fig(fig)


def radar_chart(factor_scores: dict):
    label_map = {
        "technical": "技术", "momentum": "动量", "macro": "宏观",
        "valuation": "估值", "sentiment": "情绪", "commodities": "商品",
        "event_financial_crisis": "金融危机", "event_war": "战争",
        "event_pandemic": "疫情", "event_policy": "政策",
        "event_political": "政治", "event_tech": "科技",
        "event_energy": "能源", "event_currency": "货币"
    }
    cats = [label_map.get(k, k) for k in factor_scores]
    vals = [(v + 10) for v in factor_scores.values()]  # shift to 0-20
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6, color="#60a5fa"),
        name="因子强度"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#040810",
            radialaxis=dict(visible=True, range=[0, 20],
                            gridcolor="#0f2040", tickfont_size=8),
            angularaxis=dict(gridcolor="#0f2040", tickfont_size=10)
        ),
        height=400, title="多维因子雷达图"
    )
    return style_fig(fig)


def event_impact_chart(event_scores: dict):
    types = list(event_scores.keys())
    scores = list(event_scores.values())
    colors = [EVENT_TYPES[t]["color"] for t in types]
    labels = [f"{EVENT_TYPES[t]['icon']} {EVENT_TYPES[t]['label']}" for t in types]
    fig = go.Figure(go.Bar(
        x=labels, y=scores,
        marker_color=colors, marker_opacity=0.85,
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside"
    ))
    fig.add_hline(y=0, line=dict(color="#334155", width=1.5))
    fig.update_layout(height=350, title="历史事件因子当前得分（衰减加权）",
                      yaxis=dict(title="综合影响得分", range=[-6, 6]))
    return style_fig(fig)


def events_timeline(events: list):
    if not events:
        return None
    df_e = pd.DataFrame(events)
    df_e["date"] = pd.to_datetime(df_e["date"])
    df_e["color"] = df_e["type"].map(lambda t: EVENT_TYPES.get(t, {}).get("color", "#94a3b8"))
    df_e["type_label"] = df_e["type"].map(lambda t: EVENT_TYPES.get(t, {}).get("label", t))
    df_e["size"] = df_e["decayed_impact"].abs() * 8 + 5

    fig = go.Figure()
    for et in EVENT_TYPES:
        sub = df_e[df_e["type"] == et]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["decayed_impact"],
            mode="markers+text",
            marker=dict(size=sub["size"], color=EVENT_TYPES[et]["color"],
                        line=dict(color="white", width=1), opacity=0.85),
            text=sub["label"],
            textposition="top center",
            textfont=dict(size=8),
            name=EVENT_TYPES[et]["label"],
            hovertemplate="<b>%{text}</b><br>日期: %{x|%Y-%m-%d}<br>衰减冲击: %{y:.3f}<br>",
        ))
    fig.add_hline(y=0, line=dict(color="#334155", width=1))
    fig.update_layout(height=400, title="活跃历史事件时间轴（衰减后冲击）",
                      yaxis=dict(title="衰减后冲击"), xaxis=dict(title="事件日期"))
    return style_fig(fig)


# ─────────────────────────────────────────────
# SECTION 8: MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:center">
            <div>
                <p class="hero-title">⚡ DJIA QUANTUM PREDICTOR</p>
                <p class="hero-subtitle">
                    AI多因子预测引擎 · 40+因子 · 技术/宏观/商品/历史事件 · Claude Sonnet驱动
                </p>
            </div>
            <div style="text-align:right">
                <span style="font-size:11px; color:#334155; letter-spacing:2px">
                    数据截至 2026-03-08 · 仅供研究
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    api_key, macro = render_sidebar()

    # ── Load data ──
    df = build_price_series()
    latest = df.iloc[-1]

    # ── Compute event factors (relative to latest date) ──
    ref_date = latest["date"].strftime("%Y-%m-%d")
    event_scores, active_events = compute_event_factors(ref_date)

    # ── Top metrics bar ──
    cols = st.columns(8)
    metrics = [
        ("DJIA", f"{latest['close']:,.0f}", None),
        ("RSI(14)", f"{latest['rsi']:.1f}", "超买" if latest['rsi']>70 else "超卖" if latest['rsi']<30 else "中性"),
        ("MACD", f"{latest['macd']:+.0f}", "多" if latest['macd']>0 else "空"),
        ("波动率", f"{latest['vol20']:.1f}%", None),
        ("VIX", f"{macro['vix']}", "恐慌" if macro['vix']>30 else "警惕" if macro['vix']>20 else "平静"),
        ("黄金", f"${macro['gold']:,}", None),
        ("WTI", f"${macro['wti']}", None),
        ("CAPE", f"{macro['cape']}x", "高估" if macro['cape']>30 else "合理"),
    ]
    for col, (label, val, sub) in zip(cols, metrics):
        with col:
            st.metric(label, val, sub)

    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 预测引擎", "📈 技术图表", "🌍 历史事件因子",
        "📊 因子全览", "📋 事件数据库"
    ])

    # ════════════════════════════
    # TAB 1: PREDICTION
    # ════════════════════════════
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            scenario = st.text_area(
                "💬 场景描述（可选）",
                placeholder="描述已知的近期事件，例如：'美联储明天可能降息25bp'、'非农超预期'、'中美贸易谈判破裂'...",
                height=80
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("⚡ 运行多因子预测", use_container_width=True)

        if not api_key:
            st.warning("⚠️ 请在左侧边栏输入 Anthropic API Key 后运行预测")

        if run_btn and api_key:
            with st.spinner("🔮 AI引擎正在分析40+因子..."):
                try:
                    pred = run_prediction(df, macro, event_scores, active_events, scenario, api_key)
                    st.session_state["pred"] = pred
                except Exception as e:
                    st.error(f"预测失败: {e}")

        pred = st.session_state.get("pred")

        if pred:
            direction = pred["direction"]
            dir_color = "#10b981" if direction == "BULLISH" else "#ef4444" if direction == "BEARISH" else "#f59e0b"
            dir_zh = "▲ 看多" if direction == "BULLISH" else "▼ 看空" if direction == "BEARISH" else "◆ 中性"
            ret = pred["expected_return_pct"]

            # Main prediction banner
            st.markdown(f"""
            <div class="pred-card">
              <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px; align-items:center">
                <div style="text-align:center; border-right:1px solid #0f2040">
                  <div style="font-size:11px; color:#475569; letter-spacing:2px; margin-bottom:8px">明日预测收盘</div>
                  <div style="font-size:48px; font-weight:800; color:{dir_color}">{pred['prediction']:,.2f}</div>
                  <div style="font-size:16px; color:{dir_color}; margin-top:4px">
                    {'+' if ret>0 else ''}{ret:.2f}%
                  </div>
                </div>
                <div style="text-align:center; border-right:1px solid #0f2040">
                  <div style="font-size:11px; color:#475569; letter-spacing:2px; margin-bottom:8px">80% 置信区间</div>
                  <div style="font-size:20px; font-weight:700; color:#94a3b8">
                    {pred['confidence_low']:,.0f} — {pred['confidence_high']:,.0f}
                  </div>
                  <div style="font-size:11px; color:#475569; margin-top:6px">
                    ±{((pred['confidence_high']-pred['confidence_low'])/2/pred['prediction']*100):.1f}%
                  </div>
                </div>
                <div style="text-align:center">
                  <div style="font-size:11px; color:#475569; letter-spacing:2px; margin-bottom:8px">方向 · 置信度</div>
                  <div style="font-size:24px; font-weight:800; color:{dir_color}">{dir_zh}</div>
                  <div style="font-size:13px; color:#64748b; margin-top:6px">{pred['confidence']}% confidence</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Regime + bias
            c1, c2, c3, c4 = st.columns(4)
            regime_zh = {"RISK_ON": "🟢 风险偏好", "RISK_OFF": "🔴 避险模式", "TRANSITIONAL": "🟡 过渡期"}
            bias_zh = {"BULLISH": "▲ 看多", "BEARISH": "▼ 看空", "NEUTRAL": "◆ 中性"}
            for col, (label, val) in zip([c1,c2,c3,c4], [
                ("市场体制", regime_zh.get(pred.get("regime",""), pred.get("regime",""))),
                ("周线偏向", bias_zh.get(pred.get("week_bias",""), "")),
                ("月线偏向", bias_zh.get(pred.get("month_bias",""), "")),
                ("季线偏向", bias_zh.get(pred.get("quarter_bias",""), "")),
            ]):
                col.metric(label, val)

            # Narrative
            st.markdown(f"""
            <div style="background:#080e1c; border:1px solid #0f2040; border-radius:8px;
                        padding:16px; margin:12px 0">
              <div style="font-size:10px; color:#475569; letter-spacing:2px; margin-bottom:8px">▸ 模型分析</div>
              <p style="font-size:12px; color:#94a3b8; line-height:1.8; margin:0">{pred['narrative']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Bulls / Bears / Risks
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**▲ 看多因子**")
                for f in pred.get("top_bullish", []):
                    st.markdown(f'<div class="bull-factor">{f}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**▼ 看空因子**")
                for f in pred.get("top_bearish", []):
                    st.markdown(f'<div class="bear-factor">{f}</div>', unsafe_allow_html=True)
            with c3:
                st.markdown("**⚠ 关键风险**")
                for r in pred.get("key_risks", []):
                    st.markdown(f'<div class="risk-box" style="font-size:11px;color:#92400e">{r}</div>',
                                unsafe_allow_html=True)

            # Charts
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(factor_bar_chart(pred["factor_scores"]),
                                use_container_width=True)
            with c2:
                st.plotly_chart(radar_chart(pred["factor_scores"]),
                                use_container_width=True)

    # ════════════════════════════
    # TAB 2: PRICE CHART
    # ════════════════════════════
    with tab2:
        days = st.select_slider("图表范围", [30, 60, 90, 180, 365, 999],
                                value=90,
                                format_func=lambda x: "全部" if x==999 else f"{x}日")
        pred = st.session_state.get("pred")
        st.plotly_chart(price_chart(df, pred, days), use_container_width=True)

        # Additional indicators
        c1, c2 = st.columns(2)
        with c1:
            sub = df.tail(days)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["bb_pos"],
                line=dict(color="#a78bfa", width=1.8), name="布林带位置"))
            fig.add_hline(y=0.8, line=dict(color="#ef4444", width=1, dash="dot"))
            fig.add_hline(y=0.2, line=dict(color="#10b981", width=1, dash="dot"))
            fig.update_layout(height=250, title="布林带相对位置", yaxis=dict(range=[0,1]))
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["vol20"],
                line=dict(color="#f59e0b", width=1.8), name="年化波动率(20d)"))
            fig.update_layout(height=250, title="实现波动率 (20日年化)")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    # ════════════════════════════
    # TAB 3: HISTORICAL EVENT FACTORS
    # ════════════════════════════
    with tab3:
        st.markdown(f"""
        <div style="background:#080e1c;border:1px solid #0f2040;border-radius:8px;padding:14px;margin-bottom:16px">
          <div style="font-size:11px;color:#60a5fa;letter-spacing:2px;margin-bottom:6px">▸ 历史事件因子系统说明</div>
          <p style="font-size:11px;color:#64748b;margin:0;line-height:1.7">
            本系统收录了自1918年以来全球 <b style="color:#94a3b8">{len(HISTORICAL_EVENTS)}个</b>
            重大历史事件，按9大类型分类。每个事件设有<b style="color:#94a3b8">冲击强度（-5至+5）</b>
            和<b style="color:#94a3b8">市场冲击衰减周期</b>（事件对市场的影响并非立竿见影，而是随时间指数衰减）。
            参考日期: <b style="color:#93c5fd">{ref_date}</b> | 活跃事件数: <b style="color:#93c5fd">{len(active_events)}</b>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(event_impact_chart(event_scores), use_container_width=True)

        if active_events:
            tl = events_timeline(active_events)
            if tl:
                st.plotly_chart(tl, use_container_width=True)

        # Active events table
        if active_events:
            st.markdown("**当前活跃历史事件（在衰减窗口内）**")
            df_active = pd.DataFrame(active_events)[
                ["date", "label", "type", "impact", "decay_months", "months_elapsed", "decayed_impact", "desc"]
            ].rename(columns={
                "date": "日期", "label": "事件", "type": "类型",
                "impact": "原始冲击", "decay_months": "衰减周期(月)",
                "months_elapsed": "已过(月)", "decayed_impact": "当前影响",
                "desc": "描述"
            }).sort_values("当前影响", key=abs, ascending=False)
            df_active["类型"] = df_active["类型"].map(lambda t: f"{EVENT_TYPES.get(t,{}).get('icon','')}{EVENT_TYPES.get(t,{}).get('label',t)}")
            st.dataframe(df_active, use_container_width=True, height=300)

    # ════════════════════════════
    # TAB 4: FACTOR OVERVIEW
    # ════════════════════════════
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            # Technical summary table
            tech_data = {
                "因子": ["RSI(14)", "MACD", "布林带位置", "10日动量", "5日ROC", "20日ROC",
                         "年化波动率", "ATR(14)", "vs SMA20", "vs SMA50", "vs SMA200"],
                "数值": [
                    f"{latest['rsi']:.1f}",
                    f"{latest['macd']:+.1f}",
                    f"{latest['bb_pos']:.3f}",
                    f"{latest['momentum10']:+.2f}%",
                    f"{latest['roc5']:+.2f}%",
                    f"{latest['roc20']:+.2f}%",
                    f"{latest['vol20']:.1f}%",
                    f"{latest['atr14']:.0f}pts",
                    f"{latest['price_vs_sma20']:+.2f}%",
                    f"{latest['price_vs_sma50']:+.2f}%",
                    f"{latest['price_vs_sma200']:+.2f}%",
                ],
                "信号": [
                    "超买" if latest['rsi']>70 else "超卖" if latest['rsi']<30 else "中性",
                    "多" if latest['macd']>0 else "空",
                    "高位" if latest['bb_pos']>0.8 else "低位" if latest['bb_pos']<0.2 else "中性",
                    "↑" if latest['momentum10']>0 else "↓",
                    "↑" if latest['roc5']>0 else "↓",
                    "↑" if latest['roc20']>0 else "↓",
                    "高" if latest['vol20']>25 else "低",
                    "-", "-", "-", "-"
                ]
            }
            st.markdown("**📐 技术因子**")
            st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

        with c2:
            macro_data = {
                "因子": ["联邦基金利率", "10Y国债", "2Y国债", "收益率曲线",
                        "CPI", "失业率", "GDP", "VIX", "CAPE", "黄金", "WTI油价", "DXY美元"],
                "数值": [
                    f"{macro['fed_rate']}%", f"{macro['yield_10y']}%", f"{macro['yield_2y']}%",
                    f"{macro['yield_10y']-macro['yield_2y']:+.2f}%",
                    f"{macro['cpi']}%", f"{macro['unemployment']}%", f"{macro['gdp']}%",
                    str(macro['vix']), f"{macro['cape']}x",
                    f"${macro['gold']:,}", f"${macro['wti']}", str(macro['dxy'])
                ],
                "信号": [
                    "高" if macro['fed_rate']>4 else "低",
                    "高" if macro['yield_10y']>4.5 else "正常",
                    "高" if macro['yield_2y']>4.5 else "正常",
                    "倒挂" if macro['yield_10y']<macro['yield_2y'] else "正常",
                    "高通胀" if macro['cpi']>3 else "可控",
                    "低" if macro['unemployment']<4 else "正常",
                    "强" if macro['gdp']>3 else "弱" if macro['gdp']<1 else "正常",
                    "恐慌" if macro['vix']>30 else "警惕" if macro['vix']>20 else "平静",
                    "高估" if macro['cape']>30 else "合理",
                    "高" if macro['gold']>2500 else "正常",
                    "高" if macro['wti']>90 else "正常",
                    "强" if macro['dxy']>105 else "弱",
                ]
            }
            st.markdown("**🌐 宏观 & 情绪因子**")
            st.dataframe(pd.DataFrame(macro_data), use_container_width=True, hide_index=True)

        # Event scores summary
        st.markdown("**🌍 历史事件因子得分汇总**")
        ev_df = pd.DataFrame([
            {
                "事件类型": f"{EVENT_TYPES[et]['icon']} {EVENT_TYPES[et]['label']}",
                "当前得分": f"{score:+.3f}",
                "信号方向": "📈 正面" if score > 0.1 else "📉 负面" if score < -0.1 else "➖ 中性",
                "事件总数": len([e for e in HISTORICAL_EVENTS if e["type"] == et]),
            }
            for et, score in event_scores.items()
        ])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

    # ════════════════════════════
    # TAB 5: EVENT DATABASE
    # ════════════════════════════
    with tab5:
        st.markdown(f"**历史事件数据库（共 {len(HISTORICAL_EVENTS)} 个事件）**")

        c1, c2 = st.columns([1, 3])
        with c1:
            filter_type = st.selectbox("筛选类型",
                ["全部"] + [f"{EVENT_TYPES[t]['icon']} {EVENT_TYPES[t]['label']}" for t in EVENT_TYPES])
            filter_impact = st.slider("最小冲击强度绝对值", 0.0, 5.0, 0.0, 0.5)

        df_events = pd.DataFrame(HISTORICAL_EVENTS)
        if filter_type != "全部":
            et_key = [k for k in EVENT_TYPES
                      if EVENT_TYPES[k]["label"] in filter_type][0]
            df_events = df_events[df_events["type"] == et_key]
        df_events = df_events[df_events["impact"].abs() >= filter_impact]

        df_display = df_events[["date","label","type","impact","decay_months","region","desc"]].copy()
        df_display["type"] = df_display["type"].map(
            lambda t: f"{EVENT_TYPES.get(t,{}).get('icon','')} {EVENT_TYPES.get(t,{}).get('label',t)}")
        df_display.columns = ["日期","事件名称","类型","冲击强度","衰减(月)","地区","描述"]
        df_display = df_display.sort_values("日期", ascending=False)
        st.dataframe(df_display, use_container_width=True, height=500, hide_index=True)

        # Event type distribution chart
        type_counts = pd.DataFrame(HISTORICAL_EVENTS)["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        type_counts["label"] = type_counts["type"].map(lambda t: f"{EVENT_TYPES.get(t,{}).get('icon','')} {EVENT_TYPES.get(t,{}).get('label',t)}")
        type_counts["color"] = type_counts["type"].map(lambda t: EVENT_TYPES.get(t, {}).get("color", "#94a3b8"))
        fig = go.Figure(go.Bar(
            x=type_counts["label"], y=type_counts["count"],
            marker_color=type_counts["color"], opacity=0.85,
            text=type_counts["count"], textposition="outside"
        ))
        fig.update_layout(height=300, title="各类型事件数量分布")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
      ⚠️ 免责声明：本模型仅供学习研究用途，不构成任何投资建议。股市预测存在固有不确定性，
      历史规律不保证未来表现。所有预测均基于统计模型和AI推理，实际市场受无数不可预测因素影响。
      请勿基于本工具做出实际投资决策。Market predictions are inherently uncertain.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
