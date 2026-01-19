"""
智能搜索助手 - 基于 LangGraph 的多功能 Agent 系统
具备决策能力，能自动判断是否需要搜索或直接回答
"""

import asyncio
from datetime import date, datetime
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import os
from dotenv import load_dotenv
from tavily import TavilyClient
import json
import re

# 加载环境变量
load_dotenv()

# 定义状态结构
class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    user_query: str
    next_action: str  # 下一步行动：answer_directly, search_first, use_tools
    search_needed: bool
    search_query: str
    search_results: str
    context: List[dict]
    final_answer: str
    step: str
    tool_output: str  # 工具执行结果

# 初始化模型和客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "deepseek-ai/DeepSeek-V3.2"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    temperature=0.7
)

# 初始化Tavily客户端
tavily_client = None
if os.getenv("TAVILY_API_KEY"):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ==================== 定义工具 ====================
@tool
def web_search(query: str) -> str:
    """
    使用Tavily搜索网络获取最新、最准确的信息。
    
    参数：
        query: 搜索关键词，中英文均可
        
    返回：
        格式化后的搜索结果，包含综合答案和相关链接
    """
    if not tavily_client:
        return "错误：未配置Tavily API密钥，无法进行搜索。请在.env文件中设置TAVILY_API_KEY。"
    
    try:
        print(f"🔍 执行搜索: {query}")
        
        # 添加当前日期确保时效性
        cur_date = date.today().strftime("%Y年%m月%d日")
        enhanced_query = f"{query}，当前日期是{cur_date}"
        
        # 调用Tavily搜索API
        response = tavily_client.search(
            query=enhanced_query,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            max_results=5,
            timeframe="year"
        )
        
        # 处理搜索结果
        search_results = ""
        
        # 优先使用Tavily的综合答案
        if response.get("answer"):
            search_results = f"【综合答案】\n{response['answer']}\n\n"
        
        # 添加具体的搜索结果
        if response.get("results"):
            search_results += "【相关信息】\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "无标题")
                content = result.get("content", "无内容")
                url = result.get("url", "无链接")
                # 截断过长的内容
                if len(content) > 200:
                    content = content[:200] + "..."
                search_results += f"{i}. 📰 {title}\n   📝 {content}\n   🔗 来源: {url}\n\n"
        
        if not search_results:
            search_results = "抱歉，没有找到相关信息。请尝试不同的关键词。"
        
        return search_results
        
    except Exception as e:
        return f"搜索失败: {str(e)}。请检查网络连接或API密钥。"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 清理表达式，只保留安全字符
        safe_expression = re.sub(r'[^\d\+\-\*\/\(\)\.\s]', '', expression)
        if not safe_expression.strip():
            return "错误：未提供有效的数学表达式"
        
        # 使用更安全的计算方式
        try:
            # 简单表达式计算
            result = eval(safe_expression, {"__builtins__": {}}, {})
            return f"计算结果: {safe_expression} = {result}"
        except:
            return "错误：无法计算该表达式，请确保表达式格式正确"
            
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def date_time_info(query: str = "") -> str:
    """获取当前日期和时间信息"""
    now = datetime.now()
    info = {
        "当前日期": now.strftime('%Y年%m月%d日'),
        "当前时间": now.strftime('%H:%M:%S'),
        "星期": ['一', '二', '三', '四', '五', '六', '日'][now.weekday()],
        "月份": now.strftime('%B'),
        "年份": now.year,
        "是否闰年": "是" if (now.year % 4 == 0 and now.year % 100 != 0) or (now.year % 400 == 0) else "否"
    }
    
    result = "📅 日期时间信息：\n"
    for key, value in info.items():
        result += f"  • {key}: {value}\n"
    
    return result

# ==================== 定义节点 ====================
def receive_input_node(state: AgentState) -> AgentState:
    """接收用户输入并初始化状态"""
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        user_message = state.get("user_query", "")
    updated_messages = state["messages"] + [HumanMessage(content=user_message)]
    return {
        "user_query": user_message,
        "messages": updated_messages,
        "step": "received_input"
    }

def decide_action_node(state: AgentState) -> AgentState:
    """决策节点：判断下一步行动"""
    
    user_query = state["user_query"]
    
    decision_prompt = f"""请结合历史对话和用户查询分析用户意图并决定下一步行动：

用户查询："{user_query}"
历史对话："{state["messages"]}"

请完成以下任务：
1. 分析用户意图：简洁总结用户想要了解什么
2. 决定下一步行动：
    (1)是否需要搜索最新信息？
    (2)是否需要使用特定工具？
    (3)是否可以直接回答？


- 需要搜索：涉及新闻、实时数据、最新技术、股票价格、天气预报、需要验证的信息
- 需要工具：数学计算、日期时间查询、单位换算等
- 直接回答：通用知识、定义解释、历史事实、无需实时信息的问题

特别说明：
- 如果包含"计算"、"等于"、"+"、"-"、"*"、"/"等，使用计算器
- 如果包含"时间"、"日期"、"今天"、"现在"、"星期"等，查询时间
- 如果包含"最新"、"新闻"、"股价"、"天气"、"2025"等，进行搜索

请只返回以下JSON格式：
{{
    "analysis": "用户意图",
    "next_action": "answer_directly|search_first|use_tools",
    "search_query": "如果需要搜索，生成搜索关键词",
    "reason": "决策理由",
    "tool_needed": "如果需要工具，指定工具名称：calculator|date_time|web_search"
}}"""

    try:
        response = llm.invoke([SystemMessage(content=decision_prompt)])
        
        # 尝试解析JSON
        try:
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            # 如果JSON解析失败，提取关键信息
            content = response.content
            decision = {
                "analysis": "自动分析",
                "reason": "基于关键词判断",
                "tool_needed": "none"
            }
            
            # 关键词分析
            search_keywords = ["最新", "新闻", "实时", "今天", "现在", "搜索", "查一下", 
                             "如何", "怎样", "2025", "股价", "天气", "行情", "新冠"]
            tool_keywords = ["计算", "等于", "+", "-", "*", "/", "加", "减", "乘", "除"]
            time_keywords = ["时间", "日期", "星期", "几号", "年月日"]
            
            if any(keyword in user_query.lower() for keyword in search_keywords):
                decision["next_action"] = "search_first"
                decision["search_query"] = user_query
                decision["tool_needed"] = "web_search"
            elif any(keyword in user_query.lower() for keyword in tool_keywords):
                decision["next_action"] = "use_tools"
                decision["tool_needed"] = "calculator"
            elif any(keyword in user_query.lower() for keyword in time_keywords):
                decision["next_action"] = "use_tools"
                decision["tool_needed"] = "date_time"
            else:
                decision["next_action"] = "answer_directly"
        
        # 提取决策信息
        next_action = decision.get("next_action", "answer_directly")
        search_needed = next_action in ["search_first", "use_tools"] and decision.get("tool_needed") == "web_search"
        search_query = decision.get("search_query", user_query)
        tool_needed = decision.get("tool_needed", "none")
        
        print(f"\n🤔 决策分析: {decision.get('analysis', '')}")
        print(f"📋 下一步行动: {next_action}")
        print(f"🛠️ 需要工具: {tool_needed}")
        print(f"📝 决策理由: {decision.get('reason', '')}")
        
        return {
            "next_action": next_action,
            "search_needed": search_needed,
            "search_query": search_query,
            "tool_output": tool_needed,
            "step": "decided_action",
            "messages": state["messages"] + [AIMessage(content=f"决策结果：{decision.get('analysis', '')}")]
        }
        
    except Exception as e:
        print(f"决策节点错误: {e}")
        # 默认决策
        return {
            "next_action": "answer_directly",
            "search_needed": False,
            "search_query": user_query,
            "tool_output": "none",
            "step": "decided_action",
            "messages": state["messages"] + [AIMessage(content="使用默认决策：直接回答")]
        }

def direct_answer_node(state: AgentState) -> AgentState:
    """直接回答问题（无需搜索）"""
    
    answer_prompt = f"""请直接回答用户的问题，无需搜索外部信息：

用户问题：{state['user_query']}
历史对话：{state['messages']}
要求：
1. 基于您的知识提供准确回答
2. 如果信息不足或不确定，请诚实说明
3. 回答要简洁清晰，易于理解
4. 如果问题是开放性的，提供多个角度的分析

请开始回答："""
    
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    
    return {
        "final_answer": response.content,
        "step": "answered_directly",
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

def search_node(state: AgentState) -> AgentState:
    """执行搜索"""
    
    if not state.get("search_needed", True):
        # 如果不需要搜索，跳过
        return {
            "search_results": "",
            "step": "skipped_search",
            "messages": state["messages"] + [AIMessage(content="无需搜索，直接回答")]
        }
    
    # 使用web_search工具
    try:
        search_results = web_search.invoke(state["search_query"])
        
        # 限制搜索结果的长度
        if len(search_results) > 1000:
            search_results = search_results[:1000] + "...\n(内容已截断)"
        
        return {
            "search_results": search_results,
            "step": "searched",
            "tool_output": "web_search",
            "messages": state["messages"] + [AIMessage(content="✅ 搜索完成，获取到最新信息")]
        }
    except Exception as e:
        error_msg = f"搜索执行失败: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "search_results": error_msg,
            "step": "search_failed",
            "messages": state["messages"] + [AIMessage(content="搜索遇到问题，将基于已有知识回答")]
        }

def tool_node(state: AgentState) -> AgentState:
    """执行工具调用"""
    
    tool_needed = state.get("tool_output", "none")
    user_query = state["user_query"]
    result = ""
    
    print(f"🛠️ 使用工具: {tool_needed}")
    
    try:
        if tool_needed == "calculator":
            # 提取数学表达式
            expression_match = re.search(r'[\d+\-*/().\s]+', user_query)
            if expression_match:
                expression = expression_match.group().strip()
                result = calculator.invoke(expression)
            else:
                # 尝试从文本中提取计算问题
                result = calculator.invoke(user_query)
                
        elif tool_needed == "date_time":
            result = date_time_info.invoke(user_query)
            
        elif tool_needed == "web_search":
            result = web_search.invoke(user_query)
            
        else:
            # 默认尝试搜索
            result = web_search.invoke(user_query)
            
        # 限制结果长度
        if len(result) > 800:
            result = result[:800] + "...\n(结果已截断)"
            
    except Exception as e:
        result = f"工具执行错误: {str(e)}"
    
    return {
        "search_results": result,
        "step": "tools_executed",
        "tool_output": tool_needed,
        "messages": state["messages"] + [AIMessage(content=f"工具执行完成")]
    }

def generate_final_answer_node(state: AgentState) -> AgentState:
    """生成最终答案"""
    
    # 根据不同的步骤处理
    if state["step"] == "answered_directly":
        # 直接回答的情况
        return {
            "final_answer": state.get("final_answer", "抱歉，无法回答这个问题。"),
            "step": "completed",
            "messages": state["messages"]
        }
    
    # 准备提示词
    if state.get("search_results"):
        # 有搜索结果或工具结果
        answer_prompt = f"""基于以下信息生成最终回答：

用户问题：{state['user_query']}，历史对话：{state['messages']}

{'搜索结果/工具输出：' + state['search_results'] if state['search_results'] else '无额外信息'}

要求：
1. 准确回答用户的核心问题
2. 如果使用了工具，直接给出工具的计算结果或查询结果
3. 如果是搜索信息，整合关键信息，注明来源（如果有）
4. 回答要完整、准确、有用
5. 保持友好、专业的语气

请生成最终回答："""
    else:
        # 没有额外信息
        answer_prompt = f"""请回答用户的问题：

用户问题：{state['user_query']}

请提供准确、有帮助的回答："""
    
    try:
        response = llm.invoke([SystemMessage(content=answer_prompt)])
        
        final_answer = response.content
        
        # 添加工具使用说明
        tool_used = state.get("tool_output", "")
        if tool_used and tool_used != "none":
            final_answer += f"\n\n---\nℹ️ 本次使用了 {tool_used} 工具获取信息"
        
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": state["messages"] + [AIMessage(content=final_answer)]
        }
        
    except Exception as e:
        error_msg = f"生成答案时出错: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "final_answer": f"抱歉，生成回答时遇到问题。错误信息: {str(e)}",
            "step": "error",
            "messages": state["messages"] + [AIMessage(content="生成回答时遇到问题")]
        }

def router_node(state: AgentState) -> str:
    """路由节点：根据决策结果跳转到不同分支"""
    
    next_action = state.get("next_action", "answer_directly")
    tool_needed = state.get("tool_output", "none")
    
    print(f"🔄 路由决策: {next_action}, 工具: {tool_needed}")
    
    if next_action == "answer_directly":
        return "direct_answer"
    elif next_action == "search_first":
        return "search"
    elif next_action == "use_tools":
        return "tools"
    else:
        return "direct_answer"

# ==================== 构建工作流 ====================
def create_intelligent_agent():
    """创建智能Agent工作流"""
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("receive_input", receive_input_node)
    workflow.add_node("decide_action", decide_action_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("generate_answer", generate_final_answer_node)
    
    # 设置流程
    workflow.add_edge(START, "receive_input")
    workflow.add_edge("receive_input", "decide_action")
    
    # 条件路由
    workflow.add_conditional_edges(
        "decide_action",
        router_node,
        {
            "direct_answer": "direct_answer",
            "search": "search",
            "tools": "tools"
        }
    )
    
    # 汇聚到最终答案生成
    workflow.add_edge("direct_answer", "generate_answer")
    workflow.add_edge("search", "generate_answer")
    workflow.add_edge("tools", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# ==================== 主函数 ====================
async def main():
    """运行智能Agent"""
    
    # 检查API密钥
    if not os.getenv("LLM_API_KEY"):
        print("❌ 错误：请在.env文件中配置LLM_API_KEY")
        print("   格式：LLM_API_KEY=your_api_key_here")
        return
    
    app = create_intelligent_agent()
    
    print("\n" + "=" * 60)
    print("🤖 智能助手启动！")
    print("=" * 60)
    print("我能帮您：")
    print("• 回答一般知识问题（无需搜索）")
    print("• 搜索最新信息（需要配置TAVILY_API_KEY）")
    print("• 进行数学计算：计算 25 * 4 + 100")
    print("• 查询日期时间：今天星期几？")
    print("• 获取最新新闻：今日头条新闻")
    print("=" * 60)
    print("(输入 'quit' 或 '退出' 结束对话)")
    print("=" * 60 + "\n")
    
    session_count = 0
    initial_state = {
            "messages": [],
            "user_query": "",
            "next_action": "",
            "search_needed": False,
            "search_query": "",
            "search_results": "",
            "context": [],
            "final_answer": "",
            "step": "start",
            "tool_output": ""
        }
    while True:
        user_input = input("\n💬 请输入您的问题: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("\n感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始状态
        user_query = user_input
        initial_state["user_query"] = user_query
        initial_state['messages'].append(HumanMessage(content=user_input))
        try:
            print("\n" + "=" * 60)
            
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            # 只显示关键节点的消息
                            if node_name == "generate_answer":
                                print(f"\n✨ 最终回答:\n{latest_message.content}")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"❌ 系统错误: {e}")
            print("请重新输入您的问题。\n")
'''
def test_agent():
    """测试Agent功能"""
    test_cases = [
        "计算一下 25 * 4 + 100 等于多少？",
        "今天星期几？",
        "什么是人工智能？",
        "帮我搜索最新的科技新闻",
        "2025年的春节是几月几号？"
    ]
    
    print("🧪 开始测试Agent功能...\n")
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {query}")
        print("-" * 40)
        
        # 模拟用户输入
        user_input = query
        
        # 这里可以添加测试代码来验证每个功能
        if "计算" in query:
            print("预期：使用计算器工具")
        elif "星期" in query or "时间" in query:
            print("预期：使用日期时间工具")
        elif "搜索" in query or "最新" in query:
            print("预期：使用搜索工具")
        else:
            print("预期：直接回答")
        
        print("-" * 40)
'''
if __name__ == "__main__":
    # 可以取消注释下面一行来运行测试
    # test_agent()
    
    asyncio.run(main())