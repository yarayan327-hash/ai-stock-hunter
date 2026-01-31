import streamlit as st
import pandas as pd
import pandas_ta as ta
import baostock as bs
import yfinance as yf
import requests
import json
import time
import random
from supabase import create_client
from datetime import datetime, timedelta

# ==========================================
# 0. 核心配置 & 提示词
# ==========================================
SYSTEM_PROMPT = """
你是一个资深的量化交易员，严格遵循“少妇战法”体系。
请基于传入的技术指标、资金流向和新闻，对该股票进行【买入】或【持仓】评分。

⚡ **格式要求 (关键信息必须使用背景色高亮)**:
- 关键利好/买入信号：请使用 :green-background[文字] 包裹
- 关键风险/卖出信号：请使用 :red-background[文字] 包裹
- 关键点位/支撑压力：请使用 :orange-background[文字] 包裹
- 核心结论分数：请使用 :blue-background[文字] 包裹

🔥 **买入标准 (猎手狙击)**:
1. 极致缩量 (<5日均量)。
2. 回踩生命线 (MA60) 不破。
3. J值超卖 (<20)。
4. 资金净流入或主力控盘。

请输出：
### 1. 🎯 核心结论 (评分 0-100)
### 2. 🔍 逻辑拆解 (资金/形态/指标)
### 3. 💡 操作计划 (止损位/目标位)
"""

# 美股核心池
US_CORE_POOL = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "COIN", "MSTR", "BABA", "PDD"]

st.set_page_config(page_title="市场猎手", layout="wide")

@st.cache_resource
def init_supabase():
    try: return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except: return None

def load_user_portfolio(username):
    sb = init_supabase()
    if not sb: return []
    try:
        res = sb.table("user_portfolios").select("portfolio_data").eq("username", username).execute()
        return res.data[0]['portfolio_data'] if res.data else []
    except: return []

def save_user_portfolio(username, portfolio):
    sb = init_supabase()
    if not sb: return
    try:
        existing = sb.table("user_portfolios").select("*").eq("username", username).execute()
        if existing.data:
            sb.table("user_portfolios").update({"portfolio_data": portfolio}).eq("username", username).execute()
        else:
            sb.table("user_portfolios").insert({"username": username, "portfolio_data": portfolio}).execute()
    except: pass

# ==========================================
# 1. 数据清洗
# ==========================================
def process_data(df):
    if df is None or df.empty: return None, "无数据"
    try:
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Turnover']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df = df.fillna(0)
        if 'Turnover' not in df.columns: df['Turnover'] = 0.0
            
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['MA60'] = ta.sma(df['Close'], length=60)
        kdj = ta.kdj(df['High'], df['Low'], df['Close'])
        df['J'] = kdj['J_9_3']
        df['Vol_MA5'] = ta.sma(df['Volume'], length=5)
        return df, None
    except Exception as e: return None, f"清洗失败: {str(e)}"

# ==========================================
# 2. 数据获取
# ==========================================
def get_cn_data_baostock(symbol):
    try:
        code = symbol
        if ".SS" in symbol: code = "sh." + symbol.replace(".SS", "")
        if ".SZ" in symbol: code = "sz." + symbol.replace(".SZ", "")
        if symbol.isdigit():
            code = "sh." + symbol if symbol.startswith("6") else "sz." + symbol

        bs.login()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        rs = bs.query_history_k_data_plus(code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3")
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        
        if not data_list: return None, "BaoStock无返回"
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={
            'date':'Date', 'open':'Open', 'high':'High', 
            'low':'Low', 'close':'Close', 'volume':'Volume', 
            'amount':'Turnover'
        })
        df.set_index('Date', inplace=True)
        return process_data(df)
    except Exception as e: return None, f"BS Error: {e}"

def get_hk_us_data_yf(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        if df.empty: return None, "Yahoo未返回数据"
        df['Turnover'] = df['Close'] * df['Volume']
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = df.index.tz_localize(None) 
        df.index.name = 'Date'
        return process_data(df)
    except Exception as e: return None, f"YF Error: {e}"

def get_stock_data(ticker):
    ticker = ticker.upper().strip()
    if ticker.startswith("SH.") or ticker.startswith("SZ.") or ticker.endswith(".SS") or ticker.endswith(".SZ") or (ticker.isdigit() and len(ticker)==6):
        return get_cn_data_baostock(ticker)
    else:
        return get_hk_us_data_yf(ticker)

@st.cache_data(ttl=3600)
def get_dynamic_pool(market="CN", strat="TURNOVER"):
    pool = []
    try:
        if market == "CN":
            bs.login()
            rs = bs.query_hs300_stocks()
            while (rs.error_code == '0') & rs.next():
                pool.append(rs.get_row_data()[1]) 
            bs.logout()
            if len(pool) > 15: pool = random.sample(pool, 15)
        elif market == "HK":
            pool = ["00700.HK", "03690.HK", "01810.HK", "09988.HK", "00981.HK", "02015.HK", "01024.HK", "00020.HK"]
        else:
            pool = US_CORE_POOL
        return pool
    except Exception as e: return ["ERROR", str(e)]

# ==========================================
# 4. 全能 Gemini 分析引擎 (🛡️ 穷举重试版)
# ==========================================

def list_available_models(api_key):
    """🛠️ 调试工具：列出当前 Key 可用的所有模型"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = [m['name'].replace('models/', '') for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
            return models
        else:
            return [f"Error {resp.status_code}: {resp.text}"]
    except Exception as e:
        return [f"Net Error: {str(e)}"]

def call_gemini_rest(prompt, api_key):
    # 🔴 核心修复：混合使用 1.5 (稳) 和 2.0 (新)
    # 我们按顺序尝试，直到有一个成功为止
    models_to_try = [
        "gemini-1.5-flash",       # 最稳，免费额度最高
        "gemini-1.5-pro",         # 智能，免费额度高
        "gemini-2.0-flash",       # 你列表里的，但可能429
        "gemini-2.0-flash-lite",  # 你列表里的轻量版
        "gemini-1.5-flash-latest" # 备用别名
    ]
    
    last_error = ""
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": f"你是量化专家。\n{prompt}"}]}]
        }
        
        try:
            # 缩短超时时间，快速试错
            resp = requests.post(url, headers=headers, json=data, timeout=8)
            
            if resp.status_code == 200:
                result = resp.json()
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return f"✨ **Gemini 分析** (Model: {model})\n\n{text}"
                except:
                    safety = str(result.get('promptFeedback', 'Unknown'))
                    last_error = f"Blocked: {safety}"
                    continue
            else:
                # 遇到 429 (限流) 或 404 (找不到)，直接试下一个
                last_error = f"HTTP {resp.status_code} ({model})"
                time.sleep(0.5) # 稍微停顿一下防止触发高频限制
                continue
                
        except Exception as e:
            last_error = f"Net Error: {str(e)}"
            continue

    return f"❌ **Gemini 全线忙碌**\n所有模型均尝试失败。可能Google服务暂时繁忙，请稍后再试。\n最后一次报错: {last_error}"

def analyze_stock_gemini(ticker, df, news="", holdings=None):
    latest = df.iloc[-1]
    vol_display = "0"
    if latest['Volume'] > 0:
        vol_display = f"{latest['Volume']/10000:.1f}万" if latest['Volume'] > 10000 else f"{latest['Volume']:.0f}"
    
    turnover_display = ""
    if latest['Turnover'] > 0:
        val = latest['Turnover']
        amt_亿 = val / 100000000
        turnover_display = f"成交额: {amt_亿:.2f}亿"
    
    tech = f"""
    标的: {ticker}
    现价: {latest['Close']:.2f}
    MA60: {latest['MA60']:.2f}
    J值: {latest['J']:.2f}
    成交量: {vol_display}手  {turnover_display}
    缩量状况: {'极致缩量' if latest['Volume'] < latest['Vol_MA5'] else '放量'}
    """
    
    task = "【持仓诊断】" if holdings else "【机会扫描】"
    cost = f"成本: {holdings['cost']}" if holdings else ""
    prompt = f"{SYSTEM_PROMPT}\n任务:{task}\n{tech}\n{cost}\n{news}"
    
    return call_gemini_rest(prompt, st.secrets["GEMINI_API_KEY"])

# ==========================================
# 5. 主界面
# ==========================================
def main():
    if 'current_user' not in st.session_state:
        st.title("市场猎手")
        u = st.text_input(
            "用户名", 
            placeholder="请输入您的用户名 (无需注册，任意字符即可)",
            help="如果是第一次使用，随便输一个名字，系统会自动为您创建档案。"
        )
        if st.button("登录") and u:
            st.session_state.current_user = u
            st.session_state.portfolio = load_user_portfolio(u)
            st.rerun()
        return

    with st.sidebar:
        st.header(f"👤 {st.session_state.current_user}")
        if st.button("退出"): del st.session_state.current_user; st.rerun()
        st.divider()
        with st.form("add"):
            st.write("➕ **添加自选**")
            c1, c2 = st.columns(2)
            t = c1.text_input("代码", value="sh.600519", help="A股: sh.600519 | 港股: 00700.HK | 美股: NVDA")
            c = c2.number_input("持仓成本", 0.0)
            if st.form_submit_button("加入"):
                st.session_state.portfolio.append({'ticker':t.upper(), 'name':t, 'cost':c})
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()
        
        st.divider()
        st.write("🔧 **调试工具**")
        if st.button("🔍 检测可用模型"):
            with st.spinner("正在询问 Google API ..."):
                models = list_available_models(st.secrets["GEMINI_API_KEY"])
                st.write(models)
        
        st.divider()
        st.write("📦 **持仓列表**")
        for i, p in enumerate(st.session_state.portfolio):
            c1, c2 = st.columns([0.8, 0.2])
            c1.markdown(f"**{p['ticker']}**")
            if c2.button("🗑️", key=f"d{i}"):
                st.session_state.portfolio.pop(i)
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()

    st.title("市场猎手")
    st.caption("🇨🇳 A股: BaoStock | 🌍 港美股: Yahoo | 🧠 分析核心: Gemini (混合模式)")
    
    tab1, tab2 = st.tabs(["📊 持仓体检", "🌍 机会雷达"])
    
    with tab1:
        if st.button("开始体检", type="primary"):
            bar = st.progress(0)
            for i, p in enumerate(st.session_state.portfolio):
                with st.spinner(f"Gemini 正在分析 {p['ticker']} ..."):
                    df, err = get_stock_data(p['ticker'])
                    if df is not None:
                        res = analyze_stock_gemini(p['ticker'], df, "", p)
                        with st.expander(f"📌 {p['ticker']} 诊断报告", expanded=True): st.markdown(res)
                    else:
                        st.error(f"{p['ticker']} 获取失败: {err}")
                bar.progress((i+1)/len(st.session_state.portfolio))
    
    with tab2:
        c1, c2 = st.columns(2)
        m_type = c1.selectbox("选择市场", ["CN (A股)", "HK (港股)", "US (美股)"])
        c2.selectbox("扫描战法", ["🏛️ 资金战场 (成交额 Top)", "🎢 稳健活跃 (换手率 4-10%)"])
        
        if st.button("🚀 启动扫描", type="primary"):
            with st.spinner("正在猎取核心资产..."):
                pool = get_dynamic_pool(m_type.split()[0])
            
            if pool and pool[0] == "ERROR":
                st.error(f"池子获取失败: {pool[1]}")
            else:
                st.success(f"锁定 {len(pool)} 只标的，正在计算...")
                status = st.status("正在筛选...", expanded=True)
                
                valid_stocks = []
                for t in pool:
                    df, _ = get_stock_data(t)
                    if df is not None:
                        if df.iloc[-1]['J'] < 50:
                            valid_stocks.append({'t':t, 'df':df})
                
                if not valid_stocks:
                    status.update(label="暂无极佳机会", state="error")
                else:
                    status.write(f"命中 {len(valid_stocks)} 只，Gemini 正在分析...")
                    for item in valid_stocks[:3]:
                        res = analyze_stock_gemini(item['t'], item['df'])
                        with st.expander(f"🎯 {item['t']} - 机会分析", expanded=True):
                            st.markdown(res)
                            
                    status.update(label="扫描完成", state="complete")

if __name__ == "__main__":
    main()
