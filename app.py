import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import time
from openai import OpenAI
from supabase import create_client
from datetime import datetime, timedelta

# ==========================================
# 0. 核心配置
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

# 美股核心池 (用于兜底)
US_CORE_POOL = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "COIN", "MSTR", "BABA", "PDD"]

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

st.set_page_config(page_title="全球资金流向狙击", layout="wide")

# ==========================================
# 1. 统一数据引擎 (全 AkShare 实现)
# ==========================================

# 通用数据清洗函数
def process_data(df):
    if df is None or df.empty: return None, "无数据"
    try:
        # 统一列名
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['MA60'] = ta.sma(df['Close'], length=60)
        kdj = ta.kdj(df['High'], df['Low'], df['Close'])
        df['J'] = kdj['J_9_3']
        df['Vol_MA5'] = ta.sma(df['Volume'], length=5)
        
        # 确保 Turnover 列存在 (美股可能没有，补0)
        if 'Turnover' not in df.columns:
            df['Turnover'] = 0
            
        return df, None
    except Exception as e:
        return None, str(e)

def get_data_cn(symbol):
    """A股数据获取 (东方财富)"""
    try:
        # symbol 格式: "600519.SS" -> "600519"
        code = symbol.split(".")[0]
        # 获取历史K线
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
        # 🔥 修改点：同时获取成交量和成交额
        df = df.rename(columns={
            '日期':'Date', '开盘':'Open', '收盘':'Close', 
            '最高':'High', '最低':'Low', '成交量':'Volume', 
            '成交额':'Turnover'
        })
        df.set_index('Date', inplace=True)
        return process_data(df)
    except Exception as e: return None, f"CN Error: {e}"

def get_data_hk(symbol):
    """港股数据获取 (新浪/东财)"""
    try:
        # symbol 格式: "0700.HK" -> "00700"
        code = symbol.split(".")[0].zfill(5)
        df = ak.stock_hk_daily(symbol=code, adjust="qfq")
        # 🔥 修改点：确保取前7列 (包含成交额)，防止列索引溢出
        # 通常 akshare 返回: date, open, high, low, close, volume, amount
        if df.shape[1] >= 7:
            df = df.iloc[:, :7]
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        else:
            df = df.iloc[:, :6]
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Turnover'] = 0 # 缺失补0
            
        df.set_index('Date', inplace=True)
        return process_data(df)
    except Exception as e: return None, f"HK Error: {e}"

def get_data_us(symbol):
    """美股数据获取 (新浪接口 - 国内可用)"""
    try:
        # symbol 格式: "AAPL"
        clean_sym = symbol.split(".")[0]
        # 新浪美股接口
        df = ak.stock_us_daily(symbol=clean_sym, adjust="qfq")
        df = df.rename(columns={'date':'Date', 'open':'Open', 'close':'Close', 'high':'High', 'low':'Low', 'volume':'Volume'})
        # 美股接口通常只有 Volume，没有 Turnover (Amount)，设为0
        df['Turnover'] = 0
        df.set_index('Date', inplace=True)
        return process_data(df)
    except Exception as e: return None, f"US Error: {e}"

def get_stock_data(ticker):
    """智能路由：根据代码特征自动选择国内可用的接口"""
    ticker = ticker.upper().strip()
    if ticker.endswith(".SS") or ticker.endswith(".SZ") or ticker.isdigit(): # A股逻辑
        if ticker.isdigit(): # 自动补全
            ticker = f"{ticker}.SS" if ticker.startswith("6") else f"{ticker}.SZ"
        return get_data_cn(ticker)
    elif ticker.endswith(".HK"): # 港股
        return get_data_hk(ticker)
    else: # 美股 (纯字母)
        return get_data_us(ticker)

# ==========================================
# 2. 动态榜单获取
# ==========================================
def get_dynamic_pool(market="CN", strat="TURNOVER"):
    pool = []
    try:
        if market == "CN":
            df = ak.stock_zh_a_spot_em()
            df = df[df['代码'].astype(str).str.match(r'^[036]')] # 过滤B股等
            if strat == "TURNOVER":
                # 🏛️ 资金战场
                target = df.sort_values(by="成交额", ascending=False).head(30)
            elif strat == "TURNOVER_RATE":
                # 🎢 稳健活跃 (换手率4-10%且上涨)
                mask = (df['换手率']>=4) & (df['换手率']<=10) & (df['涨跌幅']>0)
                target = df[mask].sort_values(by="换手率", ascending=False).head(30)
            else: 
                # 💰 主力扫货 (净流入)
                target = df.sort_values(by="主力净流入", ascending=False).head(30)
            
            for _, r in target.iterrows():
                suffix = ".SS" if str(r['代码']).startswith("6") else ".SZ"
                pool.append(str(r['代码']) + suffix)
                
        elif market == "HK":
            df = ak.stock_hk_spot_em()
            target = df.sort_values(by="成交额", ascending=False).head(20)
            for _, r in target.iterrows():
                pool.append(str(r['代码']) + ".HK")
                
        else: # US (美股)
            pool = US_CORE_POOL
            
        return pool
    except Exception as e: return ["ERROR", str(e)]

# ==========================================
# 3. AI 分析与新闻
# ==========================================
def analyze_with_deepseek(ticker, df, news="", holdings=None):
    latest = df.iloc[-1]
    
    # 🔥 修改点：在 Prompt 中同时体现成交量和成交额
    vol_display = f"{latest['Volume']/10000:.1f}万" if latest['Volume'] > 10000 else f"{latest['Volume']:.0f}"
    
    # 只有A股港股显示成交额，美股如果为0则不显示
    turnover_display = ""
    if latest['Turnover'] > 0:
        amt_亿 = latest['Turnover'] / 100000000
        turnover_display = f"成交额: {amt_亿:.2f}亿"
    
    tech = f"""
    标的: {ticker}
    现价: {latest['Close']:.2f}
    MA60: {latest['MA60']:.2f}
    J值: {latest['J']:.2f}
    成交量: {vol_display}手 {turnover_display}
    缩量状况: {'极致缩量' if latest['Volume'] < latest['Vol_MA5'] else '放量'}
    """
    
    task = "【持仓诊断】" if holdings else "【机会扫描】"
    cost = f"成本: {holdings['cost']}" if holdings else ""
    
    prompt = f"{SYSTEM_PROMPT}\n任务:{task}\n{tech}\n{cost}\n{news}"
    
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是量化专家。"}, {"role": "user", "content": prompt}],
            stream=False
        )
        return resp.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 4. 主界面
# ==========================================
def main():
    if 'current_user' not in st.session_state:
        st.title("🤖 DeepSeek 市场猎手 (CN专版)")
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
            t = c1.text_input("代码 (如 NVDA/00700.HK)", "600519.SS")
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

    # 🔥 修改点：主标题更新
    st.title("🌊 全球资金流向狙击 (动态数据)")
    tab1, tab2 = st.tabs(["📊 持仓体检", "🌍 机会雷达"])
    
    with tab1:
        if st.button("一键体检"):
            bar = st.progress(0)
            for i, p in enumerate(st.session_state.portfolio):
                df, err = get_stock_data(p['ticker'])
                if df is not None:
                    res = analyze_with_deepseek(p['ticker'], df, "", p)
                    with st.expander(f"📌 {p['ticker']} 诊断报告", expanded=True): st.markdown(res)
                else:
                    st.error(f"{p['ticker']} 数据获取失败: {err}")
                bar.progress((i+1)/len(st.session_state.portfolio))
    
    with tab2:
        c1, c2 = st.columns(2)
        m_type = c1.selectbox("选择市场", ["CN (A股)", "HK (港股)", "US (美股)"])
        
        # 🔥 修改点：补全三大维度，并做好映射
        strategy_map = {
            "🏛️ 资金战场 (成交额 Top)": "TURNOVER",
            "🎢 稳健活跃 (换手率 4-10%)": "TURNOVER_RATE",
            "💰 主力扫货 (净流入 Top)": "FLOW"
        }
        selected_strat = c2.selectbox("扫描战法", list(strategy_map.keys()))
        strat_code = strategy_map[selected_strat]
        
        m_code = m_type.split()[0]
        
        if st.button("🚀 启动扫描"):
            with st.spinner("正在从国内镜像获取实时数据..."):
                pool = get_dynamic_pool(m_code, strat_code)
            
            if pool and pool[0] == "ERROR":
                st.error(pool[1])
            else:
                st.success(f"已锁定 {len(pool)} 只标的，正在计算指标...")
                status = st.status("正在进行量化筛选...", expanded=True)
                
                valid_stocks = []
                for t in pool:
                    df, _ = get_stock_data(t)
                    if df is not None:
                        # 简单的缩量回调筛选
                        last = df.iloc[-1]
                        if last['J'] < 50: # J值不过热
                            valid_stocks.append({'t':t, 'df':df})
                
                if not valid_stocks:
                    status.update(label="未发现极佳机会", state="error")
                else:
                    status.write(f"筛选出 {len(valid_stocks)} 只潜力股，DeepSeek 正在研判...")
                    # 取前3个进行AI分析
                    for item in valid_stocks[:3]:
                        res = analyze_with_deepseek(item['t'], item['df'])
                        
                        # 🔥 修改点：使用 st.expander 拉齐样式
                        with st.expander(f"🎯 {item['t']} - 机会分析", expanded=True):
                            st.markdown(res)
                            
                    status.update(label="扫描完成", state="complete")

if __name__ == "__main__":
    main()
