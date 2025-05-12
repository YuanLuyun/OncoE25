import streamlit as st
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# — Version info in sidebar —
st.sidebar.markdown("---")
st.sidebar.markdown("**OncoE25** Online Tool • Version: May 2025")


st.title("Postoperative EOCRC Prediction Model (OncoE25)")
st.write("Enter the following items to display the predicted postoperative survival risk")

# —— 1. 加载数据 & 训练模型 —— #
@st.cache_data
def load_data():
    rect_df = pd.read_csv('EOCRC_rectum_top_filtered.csv')
    col_df  = pd.read_csv('EOCRC_colon_top_filtered.csv')
    return rect_df, col_df

@st.cache_resource
def train_model(df):
    y = Surv.from_dataframe('SEER cause-specific death classification',
                            'Survival months', df)
    X = df.drop(columns=['Survival months',
                         'SEER cause-specific death classification',
                         'Patient ID'] , errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=2,
        max_depth=2,
        random_state=42
    )
    rsf.fit(X_train, y_train)
    return rsf, X_train

rect_data, colon_data = load_data()
rect_model,  rect_X_train  = train_model(rect_data)
colon_model, colon_X_train = train_model(colon_data)

# —— 2. 定义所有类别选项 —— #
ordered_var_categories = {
    'T': ['Tis', 'T1', 'T2', 'T3', 'T4'],
    'N': ['N0', 'N1', 'N2'],
    'TNM Stage': ['0', 'I', 'IIA', 'IIB', 'IIC', 'IIIA', 'IIIB', 'IIIC'],
    'Grade': ['I', 'II', 'III', 'IV'],
    'CEA': ['＜5', '≥5'],
    'Median household income': [
        '<$40,000', '40,000 - $44,999', '$45,000 - $49,999',
        '$50,000 - $54,999', '$55,000 - $59,999',
        '$60,000 - $64,999', '$65,000 - $69,999',
        '$70,000 - $74,999', '$75,000 - $79,999',
        '$80,000 - $84,999', '$85,000 - $89,999',
        '$90,000 - $94,999', '$95,000 - $99,999',
        '$100,000 - $109,999', '$110,000 - $119,999',
        '$120,000+'
    ],
    'No. of resected LNs': ['0', '1-3', '≥4']
}
sex_categories = ["Female", "Male"]
marital_categories = ["Single","Married","Widowed","Divorced","Separated"]
race_categories = ["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"]
rectum_sites = ["Rectum", "Rectosigmoid Junction"]
colon_sites = ["Ascending Colon", "Sigmoid Colon", "Hepatic Flexure", "Splenic Flexure",
               "Transverse Colon", "Descending Colon", "Cecum"]
rural_urban_categories = [
    'Metropolitan counties (1 million+)',
    'Rural counties near urban areas',
    'Medium metro counties (250K to 1M)',
    'Small metro counties (<250K)',
    'Remote rural counties (not near urban areas)'
]
histology_categories = [
    "Adenocarcinoma, NOS",
    "Mucinous adenocarcinoma",
    "Villous adenocarcinoma",
    "Signet ring cell carcinoma",
    "Other",
    "Mixed adenocarcinoma"
]
systemic_seq_categories = [
    "Untreated",
    "Postoperative",
    "Preoperative",
    "Preoperative+Postoperative",
    "Sequence unknown"
]
colon_resection_types = [
    "Partial/subtotal colectomy",
    "Hemicolectomy or greater",
    "Total colectomy",
    "Colectomy plus removal of other organs"
]
rectum_resection_types = [
    "Partial proctectomy",
    "Pull-through resection WITH sphincter preservation",
    "Abdominoperineal resection or complete proctectomy",
    "PLUS partial or total removal of other organs",
    "Pelvic Exenteration"
]
perineural_categories = ["No", "Yes"]
surg_rad_seq_categories = [
    "Untreated or unknown",
    "Preoperative",
    "Postoperative",
    "Preoperative+Postoperative",
    "Sequence unknown"
]
# —— 3. 页面三列布局 —— #
col1, col2, col3 = st.columns(3)

with col1:
    primary_site = st.selectbox("Primary site",
        options=rectum_sites + colon_sites)
    sex          = st.selectbox("Sex", options=sex_categories)
    race         = st.selectbox("Race", options=race_categories)
    marital      = st.selectbox("Marital status", options=marital_categories)
    income       = st.selectbox("Median Household Income",
                options=[
                    '<$40,000', '40,000 - $44,999', '$45,000 - $49,999',
                    '$50,000 - $54,999', '$55,000 - $59,999',
                    '$60,000 - $64,999', '$65,000 - $69,999',
                    '$70,000 - $74,999', '$75,000 - $79,999',
                    '$80,000 - $84,999', '$85,000 - $89,999',
                    '$90,000 - $94,999', '$95,000 - $99,999',
                    '$100,000 - $109,999', '$110,000 - $119,999',
                    '$120,000+'
                ])
    rural_urban  = st.selectbox("Rural-Urban Continuum",
                        options=rural_urban_categories)

with col2:
    resection    = st.selectbox("Resection type",
        options=(rectum_resection_types if primary_site in rectum_sites
                 else colon_resection_types))
    histology    = st.selectbox("Histology Type",
                        options=histology_categories)
    grade          = st.selectbox("Grade", options=ordered_var_categories['Grade'])
    t            = st.selectbox("T", options=ordered_var_categories['T'])
    n            = st.selectbox("N", options=ordered_var_categories['N'])
    tnm_stage      = st.selectbox("TNM Stage", options=ordered_var_categories['TNM Stage'])

with col3:

    cea          = st.selectbox("CEA（ng/mL）",
                        options=ordered_var_categories['CEA'])
    ln_count       = st.selectbox("No. of resected LNs", options=ordered_var_categories['No. of resected LNs'])
    perineural = st.selectbox(
    "Perineural Invasion",
    options=perineural_categories
)
    tumor_deposits = st.number_input("Tumor Deposits (numeric)", min_value=0.0, step=1.0)
    systemic_seq = st.selectbox("Systemic Surgery Sequence",
                        options=systemic_seq_categories)
    surg_rad_seq = st.selectbox(
        "Surgical & Radiation Sequence",
        options=surg_rad_seq_categories
    )



# —— 4. 编码 & 预测 —— #
if st.button("Submit"):
    # 选择对应模型与训练集列
    if primary_site in rectum_sites:
        model    = rect_model
        X_train  = rect_X_train
    else:
        model    = colon_model
        X_train  = colon_X_train

    # 构造 input_data
    one_hot_map = {}
    # ordered (T, N, CEA)
    one_hot_map["T"]                   = ordered_var_categories['T'].index(t)
    one_hot_map["N"]                   = ordered_var_categories['N'].index(n)
    one_hot_map["CEA"]                 = ordered_var_categories['CEA'].index(cea)
    one_hot_map["TNM Stage"]           = ordered_var_categories['TNM Stage'].index(tnm_stage)
    one_hot_map["Grade"]               = ordered_var_categories['Grade'].index(grade)
    one_hot_map["No. of resected LNs"] = ordered_var_categories['No. of resected LNs'].index(ln_count)
    one_hot_map["Tumor Deposits"]      = tumor_deposits  # 本身就是数值

    # numeric
    one_hot_map["Tumor Deposits"] = tumor_deposits

    # 所有 one-hot 列
    def add_onehot(prefix, options, choice):
        for opt in options:
            key = f"{prefix}_{opt}"
            one_hot_map[key] = (1 if choice == opt else 0)

    add_onehot("Sex", sex_categories, sex)
    add_onehot("Race", race_categories, race)
    add_onehot("Marital status", marital_categories, marital)
    add_onehot("Primary site", rectum_sites + colon_sites, primary_site)
    add_onehot("Rural-Urban Continuum", rural_urban_categories, rural_urban)
    add_onehot("Histology Type", histology_categories, histology)
    add_onehot("Systemic.Sur.Seq", systemic_seq_categories, systemic_seq)
    if primary_site in rectum_sites:
        add_onehot("Resection type", rectum_resection_types, resection)
    else:
        add_onehot("Resection type", colon_resection_types, resection)
        # 在 add_onehot 调用之后，补两行
    add_onehot("Surg.Rad.Seq", surg_rad_seq_categories, surg_rad_seq)
    add_onehot("Perineural Invasion", perineural_categories, perineural)


    # 转 DataFrame 并对齐列
    input_df = pd.DataFrame([one_hot_map])
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    # 预测风险
    risk_score = model.predict(input_df)[0]
    cumhaz_funcs = model.predict_cumulative_hazard_function(input_df)
    chf = cumhaz_funcs[0]  # 只有一条曲线

    # 风险分层
    all_scores = model.predict(X_train)
    q1, q2 = np.percentile(all_scores, [33.33, 66.67])

    st.markdown("### Risk Stratification")
    st.write(f"Low Risk: below {q1:.4f}")
    st.write(f"Medium Risk: between {q1:.4f} and {q2:.4f}")
    st.write(f"High Risk: above {q2:.4f}")

    if risk_score < q1:
        st.markdown("<span style='color:green;'>Low Risk</span>", unsafe_allow_html=True)
    elif risk_score < q2:
        st.markdown("<span style='color:orange;'>Medium Risk</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red;'>High Risk</span>", unsafe_allow_html=True)

    # 1/3/5-year 生存率
    time_points = [12, 36, 60]
    st.markdown("### 1, 3, 5-year Survival Rates")
    for tp in time_points:
        surv_rate = 1 - float(chf(tp))
        st.write(f"{tp//12}-year: {surv_rate:.4f}")

    # 累积风险曲线
    st.markdown("### Cumulative Hazard Curve")
    fig, ax = plt.subplots()
    ax.plot(chf.x, chf.y, label="Cumulative Hazard")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Cumulative Hazard")
    ax.legend()
    st.pyplot(fig)

    # 风险矩阵
    st.markdown("### Cumulative Hazard Function Matrix")
    risk_mat = pd.DataFrame([chf.y], columns=chf.x).T
    risk_mat.index.name = "Time (month)"
    risk_mat.columns = ["Cum. Haz."]
    st.dataframe(risk_mat, width=600)
# — Version info at page bottom —
st.markdown("---")
st.caption("OncoE25 Online Tool • Version: May 2025")
