import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ------------------------------------
# APP INIT
# ------------------------------------

app = Flask(__name__, 
            static_folder="../dist", 
            static_url_path="/")
CORS(app)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "dataset.csv")

@app.route("/", defaults={"path": ""})
@app.route("/\u003cpath:path\u003e")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return app.send_static_file(path)
    return app.send_static_file("index.html")

# ------------------------------------
# GLOBAL VARIABLES
# ------------------------------------

df_global = None
le_category = LabelEncoder()
le_payment = LabelEncoder()
scaler_knn = StandardScaler()

dt_model = None
nb_model = None
knn_model = None

# ------------------------------------
# LOAD + TRAIN
# ------------------------------------

def load_data():
    global df_global, dt_model, nb_model, knn_model

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return False

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], dayfirst=True, errors='coerce')
    df = df.fillna(0)

    df['Month'] = df['Purchase_Date'].dt.month
    df['Year'] = df['Purchase_Date'].dt.year
    df['MonthStr'] = df['Purchase_Date'].dt.strftime('%Y-%m')

    df['Category_enc'] = le_category.fit_transform(df['Category'])
    df['Payment_enc'] = le_payment.fit_transform(df['Payment_Method'])

    # Train models
    X = df[['Price (Rs.)', 'Discount (%)']]
    y = df['Payment_enc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(max_depth=4)
    dt_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    X_train_scaled = scaler_knn.fit_transform(X_train)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    df_global = df
    return True

# ------------------------------------
# UPLOAD
# ------------------------------------

@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(DATA_PATH)
    load_data()
    return jsonify({"message": "Dataset uploaded & models trained"})

# ------------------------------------
# DATA (Dashboard)
# ------------------------------------

@app.route('/api/data', methods=['GET'])
def get_data():
    if df_global is None:
        return jsonify({"error": "No dataset"}), 400

    df = df_global.copy()

    cat = request.args.get('category', 'All')
    pm = request.args.get('paymentMethod', 'All')
    start = request.args.get('start')
    end = request.args.get('end')

    if cat != 'All':
        df = df[df['Category'] == cat]
    if pm != 'All':
        df = df[df['Payment_Method'] == pm]
    if start:
        df = df[df['Purchase_Date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['Purchase_Date'] <= pd.to_datetime(end)]

    # Top 5 Customers
    top_customers = df.groupby('User_ID').agg({
        'Final_Price(Rs.)': 'sum',
        'User_ID': 'count'
    }).rename(columns={'User_ID': 'transactions', 'Final_Price(Rs.)': 'totalSpent'}).sort_values(by='totalSpent', ascending=False).head(5)
    
    top_cust_list = []
    for uid, row in top_customers.iterrows():
        top_cust_list.append({
            "userId": uid,
            "transactions": int(row['transactions']),
            "totalSpent": float(row['totalSpent'])
        })

    # KPI calculations
    total_revenue = float(df['Final_Price(Rs.)'].sum())
    total_transactions = int(len(df))
    unique_users = int(df['User_ID'].nunique())
    avg_order_value = float(df['Final_Price(Rs.)'].mean()) if len(df) > 0 else 0

    # Chart data
    monthly_revenue = df.groupby('MonthStr')['Final_Price(Rs.)'].sum().sort_index()
    category_dist = df.groupby('Category')['Final_Price(Rs.)'].sum().sort_values(ascending=False)
    payment_dist = df.groupby('Payment_Method').size().sort_values(ascending=False)

    return jsonify({
        "kpis": {
            "totalRevenue": total_revenue,
            "totalTransactions": total_transactions,
            "uniqueUsers": unique_users,
            "avgOrderValue": avg_order_value
        },
        "charts": {
            "revenueByMonth": {
                "labels": monthly_revenue.index.tolist(),
                "data": monthly_revenue.values.tolist()
            },
            "categoryShare": {
                "labels": category_dist.index.tolist(),
                "data": category_dist.values.tolist()
            },
            "paymentDistribution": {
                "labels": payment_dist.index.tolist(),
                "data": payment_dist.values.tolist()
            }
        },
        "topCustomers": top_cust_list
    })

# ------------------------------------
# CHI-SQUARE
# ------------------------------------

@app.route('/api/chi-square', methods=['GET'])
def chi_square():
    if df_global is None:
        return jsonify({"error": "No dataset"}), 400

    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    df_chi = df_global.copy()
    le_local = LabelEncoder()

    # Encode categorical features for Chi-Square
    df_chi["Category_enc"] = le_local.fit_transform(df_chi["Category"])
    df_chi["Payment_enc"] = le_local.fit_transform(df_chi["Payment_Method"])

    # Features (X) and Target (y)
    X = df_chi[["Price (Rs.)", "Discount (%)", "Final_Price(Rs.)", "Payment_enc"]]
    y = df_chi["Category_enc"]

    X = X.fillna(0)

    # Scaling (Chi-Square requires non-negative values)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Chi-Square calculation
    selector = SelectKBest(score_func=chi2, k="all")
    selector.fit(X_scaled, y)

    return jsonify({
        "features": ["Price (Rs.)", "Discount (%)", "Final_Price(Rs.)", "Payment Method"],
        "chi_scores": selector.scores_.tolist(),
        "p_values": selector.pvalues_.tolist() if hasattr(selector, 'pvalues_') else []
    })

# ------------------------------------
# CLASSIFICATION
# ------------------------------------

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.json
    price = float(data['price'])
    discount = float(data['discount'])
    model_type = data.get('model', 'nb')

    input_data = np.array([[price, discount]])

    if model_type == 'dt':
        pred = dt_model.predict(input_data)
    elif model_type == 'knn':
        input_scaled = scaler_knn.transform(input_data)
        pred = knn_model.predict(input_scaled)
    else:
        pred = nb_model.predict(input_data)

    payment = le_payment.inverse_transform(pred)

    return jsonify({
        "prediction": payment[0],
        "model": model_type
    })

# ------------------------------------
# PCA
# ------------------------------------

@app.route('/api/pca', methods=['GET'])
def pca_route():
    features = df_global[['Price (Rs.)','Discount (%)','Payment_enc']]
    X_scaled = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)

    return jsonify({
        "components": comps.tolist(),
        "variance": pca.explained_variance_ratio_.tolist()
    })

# ------------------------------------
# ASSOCIATION (Apriori + FP-Growth)
# ------------------------------------

@app.route('/api/association', methods=['GET'])
def association_route():
    if df_global is None:
        return jsonify({"error": "No dataset"}), 400

    supp = float(request.args.get('support', 0.05))
    conf = float(request.args.get('confidence', 0.3))
    algo = request.args.get('algo', 'apriori') # 'apriori' or 'fpgrowth'
    group_by = request.args.get('groupby', 'User_ID') # 'User_ID' or 'Purchase_Date'

    # Grouping transactions
    if group_by == 'Purchase_Date':
        # Group by Date
        transactions = df_global.groupby('Purchase_Date')['Category'].apply(list).tolist()
    else:
        # Group by User (Default)
        transactions = df_global.groupby('User_ID')['Category'].apply(list).tolist()

    if not transactions:
        return jsonify({"error": "No transactions found"}), 400

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

    # Frequent Itemsets
    if algo == 'fpgrowth':
        freq_itemsets = fpgrowth(df_transformed, min_support=supp, use_colnames=True)
    else:
        freq_itemsets = apriori(df_transformed, min_support=supp, use_colnames=True)

    if freq_itemsets.empty:
         return jsonify({"rules": [], "frequent_itemsets": [], "nodes": [], "links": []})

    # Association Rules
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=conf)
    
    # Prepare Rules for JSON
    rules_list = []
    nodes = set()
    links = []

    if not rules.empty:
        rules['antecedents_list'] = rules['antecedents'].apply(list)
        rules['consequents_list'] = rules['consequents'].apply(list)
        
        for _, row in rules.iterrows():
            ant = row['antecedents_list']
            cons = row['consequents_list']
            rules_list.append({
                "antecedents": ant,
                "consequents": cons,
                "support": float(row['support']),
                "confidence": float(row['confidence']),
                "lift": float(row['lift'])
            })
            
            # Graph Data (simplified: first item to first item for visualization)
            if ant and cons:
                u, v = str(ant[0]), str(cons[0])
                nodes.add(u)
                nodes.add(v)
                links.append({"source": u, "target": v, "value": float(row['lift'])})

    # Prepare Frequent Itemsets for JSON
    freq_list = []
    for _, row in freq_itemsets.sort_values(by='support', ascending=False).head(10).iterrows():
        freq_list.append({
            "items": list(row['itemsets']),
            "support": float(row['support'])
        })

    return jsonify({
        "rules": rules_list,
        "frequent_itemsets": freq_list,
        "nodes": [{"id": n} for n in nodes],
        "links": links
    })

# ------------------------------------
# TIME SERIES + SMOOTHING
# ------------------------------------

@app.route('/api/time-series', methods=['GET'])
def time_series():
    if df_global is None:
        return jsonify({"error": "No dataset"}), 400

    df = df_global.copy()
    
    # 1. Filters
    start = request.args.get('start')
    end = request.args.get('end')
    if start:
        df = df[df['Purchase_Date'] >= pd.to_datetime(start, errors='coerce')]
    if end:
        df = df[df['Purchase_Date'] <= pd.to_datetime(end, errors='coerce')]

    # 2. Metric & Aggregation
    metric_type = request.args.get('metric', 'revenue') # 'revenue' or 'transactions'
    if metric_type == 'transactions':
        monthly = df.groupby('MonthStr').size().sort_index()
    else: # Default to Revenue
        monthly = df.groupby('MonthStr')['Final_Price(Rs.)'].sum().sort_index()

    if monthly.empty:
        return jsonify({"labels": [], "original": [], "smoothed": []})

    # 3. Smoothing Parameters
    method = request.args.get('method', 'none')
    window = int(request.args.get('window', 3))
    
    original = monthly.values
    
    if method == 'mean':
        smoothed = monthly.rolling(window, center=True).mean().fillna(monthly)
    elif method == 'median':
        smoothed = monthly.rolling(window, center=True).median().fillna(monthly)
    elif method == 'boundary':
        # Boundary Smoothing: Snap to min or max of the local window
        roll = monthly.rolling(window, center=True, min_periods=1)
        mins = roll.min()
        maxs = roll.max()
        smoothed_list = []
        for i in range(len(monthly)):
            val = monthly.iloc[i]
            if abs(val - mins.iloc[i]) <= abs(val - maxs.iloc[i]):
                smoothed_list.append(float(mins.iloc[i]))
            else:
                smoothed_list.append(float(maxs.iloc[i]))
        smoothed = pd.Series(smoothed_list, index=monthly.index)
    else:
        smoothed = monthly

    return jsonify({
        "labels": monthly.index.tolist(),
        "original": original.tolist(),
        "smoothed": smoothed.tolist(),
        "metric": metric_type
    })

# ------------------------------------
# TRANSFORMATION (Binning + Normalization)
# ------------------------------------

@app.route('/api/transformation', methods=['GET'])
def transformation():
    feature = df_global['Final_Price(Rs.)']

    bins = [0,100,200,300,float("inf")]
    labels = ["Low","Medium","High","Premium"]

    df_bin = df_global.copy()
    df_bin["Price_Category"] = pd.cut(feature, bins=bins, labels=labels)

    min_max = (feature - feature.min()) / (feature.max() - feature.min())
    z_score = (feature - feature.mean()) / feature.std()

    max_abs = np.abs(feature).max()
    j = len(str(int(max_abs)))
    decimal_scaling = feature / (10**j)

    return jsonify({
        "binning": df_bin["Price_Category"].value_counts().to_dict(),
        "min_max": min_max.tolist()[:20],
        "z_score": z_score.tolist()[:20],
        "decimal_scaling": decimal_scaling.tolist()[:20]
    })

# ------------------------------------

load_data()

if __name__ == "__main__":
    app.run(debug=True, port=5000)


