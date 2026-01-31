import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import baostock as bs
import time
import random
from openai import OpenAI
from supabase import create_client
from datetime import datetime, timedelta

# ==========================================
# 🛡️ 安全气囊：防崩溃导入
# ==========================================
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️ 警告: google-generativeai 库未安装，Gemini 功能将不可用。")

# ==========================================
# 0. 核心配置 & 提示词
# ==========================================
SYSTEM_PROMPT = """
你是一个资深的量化交易员，严格遵循“少妇战法”体系。
请基于传入的技术指标、资金流向和新闻，对该股票进行【买入】或【持仓】评分。

🔥 **买入标准 (猎手狙击)**:
1. 极致缩量 (<5日均量)。
2. 回踩生命线 (MA60) 不破。
3. J值超卖 (<20)。
4. 资金净流入或主力控盘。

💼 **持仓标准**:
1. 站稳 BBI/MA20。
2. 无巨量杀跌。

请输出：
### 1. 🎯 核心结论 (评分 0-100)
### 2. 🔍 逻辑拆解 (资金/形态/指标)
### 3. 💡 操作计划 (止损位/目标位)
"""

# 美股核心池
US_CORE_POOL = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "COIN", "MSTR", "BABA", "PDD"]

st.set_page_config(page_title="全球资金流向狙击", layout="wide")

# ==========================================
# 🚨 启动检查 (如果缺库，在网页报警)
# ==========================================
if not HAS_GEMINI:
    st.warning("⚠️ 检测到服务器缺少 `google-generativeai` 库。请检查 GitHub 的 `requirements.txt` 文件是否包含该库。目前仅 A 股功能可用。")

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
        # 🛡️ 强力清洗：防止字符串导致的崩溃
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
    """A股 - BaoStock"""
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

def get_hk_us_data(ticker):
    """港美股 - AkShare"""
    try:
        ticker = ticker.upper()
        if ticker.endswith(".HK"):
            code = ticker.split(".")[0].zfill(5)
            df = ak.stock_hk_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
            if '成交额' in df.columns: df = df.rename(columns={'成交额':'Turnover'})
            else: df['Turnover'] = 0.0
        else:
            clean_sym = ticker.split(".")[0]
            df = ak.stock_us_daily(symbol=clean_sym, adjust="qfq")
            df['Turnover'] = 0.0 

        rename_map = {
            '日期':'Date', 'date':'Date', 
            '开盘':'Open', 'open':'Open', 
            '收盘':'Close', 'close':'Close', 
            '最高':'High', 'high':'High', 
            '最低':'Low', 'low':'Low', 
            '成交量':'Volume', 'volume':'Volume'
        }
        df = df.rename(columns=rename_map)
        df.set_index('Date', inplace=True)
        return process_data(df)
    except Exception as e: return None, f"接口受限: {e}"

def get_stock_data(ticker):
    ticker = ticker.upper().strip()
    if ticker.startswith("SH.") or ticker.startswith("SZ.") or ticker.endswith(".SS") or ticker.endswith(".SZ") or (ticker.isdigit() and len(ticker)==6):
        return get_cn_data_baostock(ticker)
    else:
        return get_hk_us_data(ticker)

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
            df = ak.stock_hk_spot_em()
            target = df.sort_values(by="成交额", ascending=False).head(15)
            for _, r in target.iterrows(): pool.append(str(r['代码']) + ".HK")
        else:
            pool = US_CORE_POOL
        return pool
    except Exception as e: return ["ERROR", str(e)]

# ==========================================
# 4. 双模 AI 分析引擎
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
        # 🟢 改回 gemini-pro 保证兼容性
        model = genai.GenerativeModel('gemini-pro') 
        response = model.generate_content(f"你是量化专家。\n{prompt}")
        return f"✨ **Gemini 分析 (Global)**\n\n{response.text}"
    except Exception as e: return f"Gemini Error: {e}"

def analyze_stock_router(ticker, df, news="", holdings=None):
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
    
    ticker = ticker.upper()
    is_cn = ticker.startswith("SH.") or ticker.startswith("SZ.") or ticker.endswith(".SS") or ticker.endswith(".SZ")
    
    if is_cn:
        return call_deepseek_api(prompt)
    else:
        return call_gemini_api(prompt)

# ==========================================
# 5. 主界面
# ==========================================
def main():
    if 'current_user' not in st.session_state:
        st.title("🤖 市场猎手 (DeepSeek x Gemini)")
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
            c1, c2 = st.columns(2)
            t = c1.text_input("代码 (sh.600519/AAPL)", "sh.600519")
            c = c2.number_input("成本", 0.0)
            if st.form_submit_button("加仓"):
                st.session_state.portfolio.append({'ticker':t.upper(), 'name':t, 'cost':c})
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()
        
        st.write("📦 持仓列表")
        for i, p in enumerate(st.session_state.portfolio):
            c1, c2 = st.columns([0.8, 0.2])
            c1.caption(f"{p['ticker']}")
            if c2.button("✖", key=f"d{i}"):
                st.session_state.portfolio.pop(i)
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()

    st.title("🌊 全球资金流向狙击 (双引擎版)")
    st.caption("🇨🇳 A股核心: DeepSeek | 🌍 全球市场: Google Gemini")
    
    tab1, tab2 = st.tabs(["📊 持仓体检", "🌍 机会雷达"])
    
    with tab1:
        if st.button("一键体检"):
            bar = st.progress(0)
            for i, p in enumerate(st.session_state.portfolio):
                df, err = get_stock_data(p['ticker'])
                if df is not None:
                    res = analyze_stock_router(p['ticker'], df, "", p)
                    with st.expander(f"📌 {p['ticker']} 诊断报告", expanded=True): st.markdown(res)
                else:
                    st.error(f"{p['ticker']} 失败: {err}")
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
        
        if st.button("🚀 启动扫描"):
            with st.spinner("正在获取核心资产数据..."):
                pool = get_dynamic_pool(m_type.split()[0], strat_code)
            
            if pool and pool[0] == "ERROR":
                st.error(f"数据源失败: {pool[1]}")
            else:
                st.success(f"已锁定 {len(pool)} 只核心标的，正在计算指标...")
                status = st.status("正在进行量化筛选...", expanded=True)
                
                valid_stocks = []
                for t in pool:
                    df, _ = get_stock_data(t)
                    if df is not None:
                        if df.iloc[-1]['J'] < 50:
                            valid_stocks.append({'t':t, 'df':df})
                
                if not valid_stocks:
                    status.update(label="本次抽样未发现极佳机会", state="error")
                else:
                    status.write(f"筛选出 {len(valid_stocks)} 只潜力股，AI 正在研判...")
                    for item in valid_stocks[:3]:
                        res = analyze_stock_router(item['t'], item['df'])
                        with st.expander(f"🎯 {item['t']} - 机会分析", expanded=True):
                            st.markdown(res)
                            
                    status.update(label="扫描完成", state="complete")

if __name__ == "__main__":
    main()
