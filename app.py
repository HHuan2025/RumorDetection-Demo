import streamlit as st
from bert_rumor_predict import predict_rumor
from llm_utils import call_deepseek
from triple_extraction import extract_triples
from fact_checking import fact_check
import graphviz
import paddle
from lstm_predict import load_lstm_model, evaluate_lstm

st.set_page_config(page_title="谣言检测与解释 Demo", layout="wide")
st.title("🕵️‍♂️ 谣言检测与解释 Demo")

text = st.text_area("输入文本：", height=120)

# LSTM模型评估入口
with st.expander("LSTM模型评估（需准备好评估数据eval_loader）"):
    if st.button("运行LSTM模型评估"):
        model = load_lstm_model()
        st.info("请在evaluate_lstm中传入实际的eval_loader进行评估！")
        # 示例：avg_acc, avg_loss = evaluate_lstm(model, eval_loader)
        # st.write(f"LSTM评估准确率: {avg_acc:.5f}, 损失: {avg_loss:.5f}")

if st.button("开始检测"):
    if not text.strip():
        st.warning("请输入文本！")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label, conf = predict_rumor(text)
            st.markdown(f"#### 谣言判断\n<span style='color: {'red' if label=='谣言' else 'green'}'>{label}</span><br>置信度: {conf:.2f}", unsafe_allow_html=True)
        with col2:
            explain_prompt = f"请解释为什么以下内容被判定为谣言或非谣言，并给出简要理由：\n{text}"
            explanation = call_deepseek(explain_prompt)
            st.markdown(f"#### 谣言解释\n{explanation}", unsafe_allow_html=True)
        with col3:
            triples = extract_triples(text)
            st.markdown("#### 三元组抽取")
            st.write(triples)
            # 简单知识图谱可视化
            import re
            triples_list = re.findall(r'[\(（](.*?),(.*?),(.*?)[\)）]', triples) if isinstance(triples, str) else []
            if triples_list:
                dot = graphviz.Digraph()
                for s, p, o in triples_list:
                    dot.node(s.strip())
                    dot.node(o.strip())
                    dot.edge(s.strip(), o.strip(), label=p.strip())
                st.graphviz_chart(dot)
        with col4:
            fact_result = fact_check(text)
            st.markdown(f"#### 事实核查\n{fact_result}", unsafe_allow_html=True)
