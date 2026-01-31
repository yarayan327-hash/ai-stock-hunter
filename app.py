import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import baostock as bs
import yfinance as yf  # 🟢 新引入：全球数据救星
import time
import random
from openai import OpenAI
from supabase import create_client
from datetime import datetime, timedelta

# ==========================================
# 🛡️ 防崩溃导入
# ==========================================
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==========================================
# 0. 核心配置 & 提示词 (🎨 已增加颜色指令)
# ==========================================
SYSTEM_PROMPT = """
你是一个资深的量化交易员，严格遵循“少妇战法”体系。
请基于传入的技术指标、资金流向和新闻，对该股票进行【买入】或【持仓】评分。

⚡ **格式要求 (关键信息必须染色)**:
- 关键利好/买入信号：请使用 :green[文字] 包裹 (例如 :green[资金净流入])
- 关键风险/卖出信号：请使用 :red[文字] 包裹 (例如 :red[顶部背离])
- 关键点位/支撑压力：请使用 :orange[文字] 包裹 (例如 :orange[支撑位 20.5])
- 核心结论分数：请使用 :blue[文字] 包裹

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

if not HAS_GEMINI:
    st.warning("⚠️ 警告：服务器缺少 `google-generativeai` 库，港美股AI分析可能受限。")

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
        # 强制类型转换，防止报错
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
# 2. 数据获取 (BaoStock + YFinance)
# ==========================================

def get_cn_data_baostock(symbol):
    """A股 - BaoStock (抗封锁)"""
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
    """港美股 - YFinance (雅虎财经，解决 RemoteDisconnected)"""
    try:
        # yfinance 不需要 .SS/.SZ，但港股需要 .HK
        # 如果是美股直接输代码 (NVDA)，港股输 (0700.HK)
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df.empty: return None, "Yahoo未返回数据"
        
        # yfinance 列名自带: Open, High, Low, Close, Volume
        # 需要手动处理 Turnover (yfinance 通常没有成交额，需要估算或置0)
        df['Turnover'] = df['Close'] * df['Volume'] # 估算成交额
        
        # 展平列名 (防止多级索引)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        
        # 只有日期索引需要处理一下时区
        df.index = df.index.tz_localize(None) 
        df.index.name = 'Date'
        
        return process_data(df)
    except Exception as e: return None, f"YF Error: {e}"

def get_stock_data(ticker):
    ticker = ticker.upper().strip()
    # A股特征
    if ticker.startswith("SH.") or ticker.startswith("SZ.") or ticker.endswith(".SS") or ticker.endswith(".SZ") or (ticker.isdigit() and len(ticker)==6):
        return get_cn_data_baostock(ticker)
    # 其他走 YFinance
    else:
        return get_hk_us_data_yf(ticker)

# ==========================================
# 3. 榜单获取
# ==========================================
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
            # 港股榜单依然尝试 AkShare，如果失败则返回静态池
            try:
                df = ak.stock_hk_spot_em()
                target = df.sort_values(by="成交额", ascending=False).head(15)
                for _, r in target.iterrows(): pool.append(str(r['代码']) + ".HK")
            except:
                pool = ["00700.HK", "03690.HK", "01810.HK", "09988.HK", "00981.HK"] # 兜底
        else:
            pool = US_CORE_POOL
        return pool
    except Exception as e: return ["ERROR", str(e)]

# ==========================================
# 4. 双模 AI 分析 (Gemini 修复版)
# ==========================================

def call_deepseek_api(prompt):
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是量化专家。"}, {"role": "user", "content": prompt}],
            stream=False
        )
        return f"🤖 **DeepSeek 分析 (CN)**\n\n{resp.choices[0].message.content}"
    except Exception as e: return f"DeepSeek Error: {e}"

def call_gemini_api(prompt):
    if not HAS_GEMINI:
        return "❌ 错误: Gemini 库未安装，无法分析港美股。"
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # 🟢 关键修复：指定 gemini-1.5-flash，这是目前最通用的免费模型
        model = genai.GenerativeModel('gemini-1.5-flash') 
        response = model.generate_content(f"你是量化专家。\n{prompt}")
        return f"✨ **Gemini 分析 (Global)**\n\n{response.text}"
    except Exception as e: 
        return f"Gemini Error: {e}"

def analyze_stock_router(ticker, df, news="", holdings=None):
    latest = df.iloc[-1]
    
    vol_display = "0"
    if latest['Volume'] > 0:
        vol_display = f"{latest['Volume']/10000:.1f}万" if latest['Volume'] > 10000 else f"{latest['Volume']:.0f}"
    
    turnover_display = ""
    if latest['Turnover'] > 0:
        # A股BaoStock单位是元，YFinance估算也是元
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
    
    ticker = ticker.upper()
    is_cn = ticker.startswith("SH.") or ticker.startswith("SZ.") or ticker.endswith(".SS") or ticker.endswith(".SZ")
    
    if is_cn:
        return call_deepseek_api(prompt)
    else:
        return call_gemini_api(prompt)

# ==========================================
# 5. 主界面 (🎨 UI 净化版)
# ==========================================
def main():
    if 'current_user' not in st.session_state:
        # ① UI调整：纯净标题
        st.title("市场猎手")
        u = st.text_input("用户名")
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
            # ② UI调整：详细输入指引
            c1, c2 = st.columns(2)
            t = c1.text_input(
                "股票代码", 
                value="sh.600519",
                help="🇨🇳 A股: sh.600519\n🇭🇰 港股: 00700.HK\n🇺🇸 美股: NVDA"
            )
            c = c2.number_input("持仓成本", 0.0)
            st.caption("A股: sh.600519 | 港股: 00700.HK | 美股: NVDA")
            
            if st.form_submit_button("加入"):
                st.session_state.portfolio.append({'ticker':t.upper(), 'name':t, 'cost':c})
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()
        
        st.divider()
        st.write("📦 **持仓列表**")
        for i, p in enumerate(st.session_state.portfolio):
            c1, c2 = st.columns([0.8, 0.2])
            c1.markdown(f"**{p['ticker']}**")
            if c2.button("🗑️", key=f"d{i}"):
                st.session_state.portfolio.pop(i)
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()

    # ① UI调整：主标题
    st.title("市场猎手")
    st.caption("🇨🇳 A股: BaoStock | 🌍 港美股: Yahoo Finance (稳)")
    
    tab1, tab2 = st.tabs(["📊 持仓体检", "🌍 机会雷达"])
    
    with tab1:
        if st.button("开始体检", type="primary"):
            bar = st.progress(0)
            for i, p in enumerate(st.session_state.portfolio):
                # ③ 数据与AI分析
                df, err = get_stock_data(p['ticker'])
                if df is not None:
                    res = analyze_stock_router(p['ticker'], df, "", p)
                    with st.expander(f"📌 {p['ticker']} 诊断报告", expanded=True): st.markdown(res)
                else:
                    st.error(f"{p['ticker']} 获取失败: {err}")
                bar.progress((i+1)/len(st.session_state.portfolio))
    
    with tab2:
        c1, c2 = st.columns(2)
        m_type = c1.selectbox("选择市场", ["CN (A股)", "HK (港股)", "US (美股)"])
        
        strategy_map = {
            "🏛️ 资金战场 (成交额 Top)": "TURNOVER",
            "🎢 稳健活跃 (换手率 4-10%)": "TURNOVER_RATE",
            "💰 主力扫货 (净流入 Top)": "FLOW"
        }
        selected_strat = c2.selectbox("扫描战法", list(strategy_map.keys()))
        strat_code = strategy_map[selected_strat]
        
        if st.button("🚀 启动扫描", type="primary"):
            with st.spinner("正在猎取核心资产..."):
                pool = get_dynamic_pool(m_type.split()[0], strat_code)
            
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
                    status.write(f"命中 {len(valid_stocks)} 只，AI 正在分析...")
                    for item in valid_stocks[:3]:
                        res = analyze_stock_router(item['t'], item['df'])
                        with st.expander(f"🎯 {item['t']} - 机会分析", expanded=True):
                            st.markdown(res)
                            
                    status.update(label="扫描完成", state="complete")

if __name__ == "__main__":
    main()
